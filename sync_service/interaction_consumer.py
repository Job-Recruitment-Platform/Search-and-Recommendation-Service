import logging
import hashlib
import redis
import psycopg2
import psycopg2.extras
from typing import Optional
from datetime import datetime
from app.config import Config
from models.event import InteractionEvent

logger = logging.getLogger(__name__)


class InteractionConsumer:
    """Consumer for 'user-interactions' stream - Stores interactions for CF model"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.stream_name = "user-interactions"
        self.consumer_group = "recommend-service-group"
        self.consumer_name = "python-recommend-worker-1"

        self.running = False
        self._setup_consumer_group()

    def _setup_consumer_group(self):
        """Setup consumer group for user-interactions stream"""
        try:
            try:
                stream_info = self.redis_client.xinfo_stream(self.stream_name)
                logger.info(
                    f"✓ Stream '{self.stream_name}' exists with {stream_info.get('length', 0)} messages"
                )
            except redis.exceptions.ResponseError:
                logger.warning(
                    f"⚠️  Stream '{self.stream_name}' does not exist yet"
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

    def _get_db_connection(self):
        """Get PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                database=Config.POSTGRES_DB,
                user=Config.POSTGRES_USERNAME,
                password=Config.POSTGRES_PASSWORD
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _save_interaction(self, event: InteractionEvent) -> bool:
        """Save interaction to database (for CF model training)"""
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            external_id = hashlib.md5(
                f"{event.account_id}_{event.job_id}_{event.event_type.value}_{event.occurred_at}".encode()
            ).hexdigest()[:64]

            # Insert into user_interactions table
            query = """
                INSERT INTO user_interactions 
                (account_id, job_id, event_type, occurred_at, metadata, external_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (external_id) 
                DO NOTHING
            """

            cursor.execute(query, (
                event.account_id,
                event.job_id,
                event.event_type.value,
                event.occurred_at,
                psycopg2.extras.Json(
                    event.metadata) if event.metadata else None,
                external_id
            ))

            conn.commit()
            rows_inserted = cursor.rowcount
            cursor.close()

            return rows_inserted > 0

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def process_messages(self, count: int = 10, block: int = 5000):
        """Read and process messages from user-interactions stream"""
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
                        logger.debug(f"📨 Processing interaction: {message_id}")

                        # Parse interaction event
                        event = InteractionEvent.from_redis_fields(fields)

                        logger.info(
                            f"👤 User {event.account_id} -> Job {event.job_id}: "
                            f"{event.event_type.value} (weight={event.get_weight()})"
                        )

                        # Save to database
                        saved = self._save_interaction(event)

                        if saved:
                            logger.debug(
                                f"✅ Saved interaction {message_id} to database")
                        else:
                            logger.debug(
                                f"⏭️  Skipped duplicate interaction {message_id}")

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
            f"🚀 Starting Interaction Consumer (stream: {self.stream_name})...")

        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                processed = self.process_messages()
                if processed > 0:
                    logger.info(f"✅ Processed {processed} interactions")
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
        logger.info("Stopping Interaction Consumer...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()
