import logging
import redis
import os
import socket
import time
from typing import Optional
from app.config import Config
from sync_service.sync_processor import SyncProcessor
from services.milvus_service import MilvusService

logger = logging.getLogger(__name__)


class OutboxEventConsumer:
    """Consumer for 'outbox-events' stream - Syncs Job data to Milvus"""

    def __init__(self, milvus_service: Optional[MilvusService] = None):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.stream_name = "outbox-events"
        self.consumer_group = "outbox-processor-group"
        self.consumer_name = f"python-sync-{socket.gethostname()}-{os.getpid()}"
        
        # Initialize Milvus service and sync processor
        self.milvus_service = milvus_service or MilvusService()
        self.sync_processor = SyncProcessor(self.milvus_service)

        self.running = False
        self._setup_consumer_group()
        
        self.max_retries = 5
        self.claim_idle_ms = 60000  # 1 minute
        self.retry_key_prefix = "sync-retry:"

    def _setup_consumer_group(self):
        """Setup consumer group for outbox-events stream"""
        try:
            try:
                stream_info = self.redis_client.xinfo_stream(self.stream_name)
                logger.info(
                    f"✓ Stream '{self.stream_name}' exists with {stream_info.get('length', 0)} messages"
                )
            except redis.exceptions.ResponseError:
                logger.warning(
                    f"⚠️  Stream '{self.stream_name}' does not exist yet. "
                    f"It will be created when first message arrives."
                )

            self.redis_client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0-0",
                mkstream=True,
            )
            logger.info(
                f"✓ Created consumer group '{self.consumer_group}' for stream '{self.stream_name}'"
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"✓ Consumer group '{self.consumer_group}' already exists"
                )
            else:
                logger.error(f"❌ Failed to create consumer group: {e}")
                raise
    
    # Retry helpers        
    def _retry_key(self, message_id: str) -> str:
        return f"{self.retry_key_prefix}{self.stream_name}:{message_id}"

    def _increment_retry(self, message_id: str) -> int:
        key = self._retry_key(message_id)
        retry = int(self.redis_client.incr(key))
        self.redis_client.expire(key, 7 * 24 * 3600)
        return retry

    def _clear_retry(self, message_id: str) -> None:
        self.redis_client.delete(self._retry_key(message_id))

    # Pending handling (PEL)
    def _claim_pending(self, count: int) -> list[tuple[str, dict]]:
        """
        Claim pending messages that have been idle for >= claim_idle_ms.
        Returns list of (message_id, fields).
        """
        try:
            if hasattr(self.redis_client, "xautoclaim"):
                resp = self.redis_client.xautoclaim(
                    name=self.stream_name,
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    min_idle_time=self.claim_idle_ms,
                    start_id="0-0",
                    count=count,
                )

                # redis-py may return 2 or 3 values depending on version:
                # (next_start_id, messages) OR (next_start_id, messages, deleted_ids)
                if isinstance(resp, (list, tuple)) and len(resp) == 3:
                    next_start_id, messages, deleted_ids = resp
                else:
                    next_start_id, messages = resp
                    deleted_ids = []

                if messages:
                    logger.info(f"↩️ Claimed {len(messages)} pending messages (XAUTOCLAIM)")
                    return messages
                return []

        except Exception as e:
            logger.warning(f"⚠️ XAUTOCLAIM failed/unsupported, fallback to XCLAIM: {e}")

        # Fallback: XPENDING RANGE + XCLAIM
        try:
            pendings = self.redis_client.xpending_range(
                name=self.stream_name,
                groupname=self.consumer_group,
                min="-",
                max="+",
                count=count,
            )
            if not pendings:
                return []

            ids_to_claim = [
                p["message_id"] for p in pendings if p.get("idle", 0) >= self.claim_idle_ms
            ]
            if not ids_to_claim:
                return []

            claimed = self.redis_client.xclaim(
                name=self.stream_name,
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                min_idle_time=self.claim_idle_ms,
                message_ids=ids_to_claim,
            )
            if claimed:
                logger.info(f"↩️ Claimed {len(claimed)} pending messages (XCLAIM fallback)")
            return claimed or []
        except Exception as e:
            logger.error(f"❌ Failed to claim pending messages: {e}")
            return []

    def _process_one(self, message_id: str, fields: dict) -> None:
        """Process one message. Raise on error."""
        logger.info(f"📨 Processing outbox event: {message_id}")

        result = self.sync_processor.process_stream_message(fields)

        if getattr(result, "error", None):
            raise RuntimeError(str(result.error))

        logger.info(
            f"✅ Processed {message_id}: inserted={getattr(result, 'inserted', 0)}, deleted={getattr(result, 'deleted', 0)}"
        )

    def _handle_entries(self, entries: list[tuple[str, dict]]) -> int:
        processed_count = 0

        for message_id, fields in entries:
            try:
                self._process_one(message_id, fields)

                # ✅ ACK only on success
                self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
                self._clear_retry(message_id)
                processed_count += 1

            except Exception as e:
                retry = self._increment_retry(message_id)
                logger.exception(f"❌ Error processing {message_id} (retry {retry}/{self.max_retries}): {e}")
                if retry >= self.max_retries:
                    logger.error(
                        f"🧨 Message {message_id} exceeded max_retries={self.max_retries}. "
                        f"It will remain pending (no ACK). Investigate manually."
                    )
        return processed_count

    def process_messages(self, count: int = 10, block: int = 5000) -> int:
        """Read and process messages from outbox-events stream"""
        try:
            # Reclaim & retry pending messages first
            claimed = self._claim_pending(count=count)
            if claimed:
                return self._handle_entries(claimed)

            # Read new messages
            messages = self.redis_client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block,
            )

            if not messages:
                return 0

            processed = 0
            for _stream, message_list in messages:
                processed += self._handle_entries(message_list)

            return processed

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error processing messages: {e}")
            return 0


    def run(self):
        """Run consumer continuously"""
        self.running = True
        logger.info(
            f"🚀 Starting Outbox Event Consumer "
            f"(stream: {self.stream_name}, group: {self.consumer_group}, consumer: {self.consumer_name})..."
        )

        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                processed = self.process_messages()
                if processed > 0:
                    logger.info(f"✅ Processed {processed} outbox events")
                retry_count = 0

            except KeyboardInterrupt:
                logger.info("⚠️  Consumer interrupted by user")
                break

            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Consumer error (retry {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    logger.error("❌ Max retries reached. Stopping consumer.")
                    break
                time.sleep(5 * retry_count)

        logger.info("Consumer stopped")

    def stop(self):
        """Stop consumer"""
        logger.info("Stopping Outbox Event Consumer...")
        self.running = False
        try:
            if self.redis_client:
                self.redis_client.close()
        except Exception:
            pass