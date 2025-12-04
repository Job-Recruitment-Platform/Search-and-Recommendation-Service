import logging
import redis
from typing import Optional
from app.config import Config
from sync_service.sync_processor import SyncProcessor
from services.milvus_service import MilvusService

logger = logging.getLogger(__name__)


class OutboxEventConsumer:
    """Consumer for 'outbox-events' stream - Syncs Job data to Milvus"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.stream_name = "outbox-events"
        self.consumer_group = "outbox-processor-group"
        self.consumer_name = "python-sync-worker-1"

        self.milvus_service = MilvusService()
        self.sync_processor = SyncProcessor(self.milvus_service)
        self.running = False
        self._setup_consumer_group()

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
                id="0",
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

    def process_messages(self, count: int = 10, block: int = 5000):
        """Read and process messages from outbox-events stream"""
        try:
            messages = self.redis_client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block,
            )

            if not messages:
                return 0

            processed_count = 0
            for stream, message_list in messages:
                for message_id, fields in message_list:
                    try:
                        logger.info(f"📨 Processing outbox event: {message_id}")
                        result = self.sync_processor.process_stream_message(
                            fields)

                        if result.error:
                            logger.error(
                                f"❌ Failed to process {message_id}: {result.error}"
                            )
                        else:
                            logger.info(
                                f"✅ Processed {message_id}: "
                                f"inserted={result.inserted}, deleted={result.deleted}"
                            )

                        # ACK message
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id)
                        processed_count += 1

                    except Exception as e:
                        logger.exception(
                            f"❌ Error processing {message_id}: {e}")
                        # Still ACK to avoid infinite retries
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id)

            return processed_count

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
            f"🚀 Starting Outbox Event Consumer (stream: {self.stream_name})...")

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
                logger.error(
                    f"❌ Consumer error (retry {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    logger.error("❌ Max retries reached. Stopping consumer.")
                    break
                import time
                time.sleep(5 * retry_count)

        logger.info("Consumer stopped")

    def stop(self):
        """Stop consumer"""
        logger.info("Stopping Outbox Event Consumer...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()
