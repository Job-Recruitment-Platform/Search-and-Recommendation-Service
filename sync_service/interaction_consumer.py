import logging
import json
import redis
from datetime import datetime, timezone
from collections import defaultdict
from app.config import Config, INTERACTION_WEIGHTS
from models.event import InteractionEvent

logger = logging.getLogger(__name__)

class InteractionConsumer:
    """Consumer for 'user-interactions' stream"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.stream_name = Config.INTERACTION_STREAM_NAME
        self.consumer_group = Config.INTERACTION_CONSUMER_GROUP
        self.consumer_name = Config.INTERACTION_CONSUMER_NAME

        self.running = False
        self._setup_consumer_group()

    def _setup_consumer_group(self):
        """Setup consumer group for user-interactions stream"""
        try:
            try:
                stream_info = self.redis_client.xinfo_stream(self.stream_name)
                logger.info(
                    f"Stream '{self.stream_name}' exists with {stream_info.get('length', 0)} messages"
                )
            except redis.exceptions.ResponseError:
                logger.warning(
                    f"Stream '{self.stream_name}' does not exist yet")

            self.redis_client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group '{self.consumer_group}'")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"Consumer group '{self.consumer_group}' already exists")
            else:
                logger.error(f"Failed to create consumer group: {e}")
                raise

    def _update_redis_cache(self, event: InteractionEvent) -> bool:
        try:
            cache_key = f"user_interactions:{event.account_id}"

            # Parse timestamp
            timestamp = datetime.fromisoformat(
                event.occurred_at.replace('Z', '+00:00')
            ).timestamp()

            # Get existing interactions
            interactions = defaultdict(dict)
            cached = self.redis_client.get(cache_key)
            if cached:
                try:
                    existing = json.loads(cached)
                    logger.debug(f"Loaded existing interactions for user {event.account_id}: {len(existing)} types")
                    for key, value in existing.items():
                        if isinstance(value, dict):
                            interactions[key] = value
                        elif isinstance(value, list):
                            # Convert old list format to dict
                            interactions[key] = {int(v): None for v in value}
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in cache for user {event.account_id}")
            else:
                logger.debug(f"No existing interactions for user {event.account_id}")

            # Add new interaction
            event_type_upper = event.event_type.value.upper()
            if event_type_upper in INTERACTION_WEIGHTS:
                interactions[event_type_upper][event.job_id] = timestamp
                logger.info(
                    f"✓ Added interaction: user={event.account_id}, job={event.job_id}, "
                    f"type={event_type_upper}, timestamp={timestamp}"
                )
            else:
                logger.warning(f"Event type {event_type_upper} not in INTERACTION_WEIGHTS")

            # Save back to Redis (TTL 30 days)
            final_data = {k: dict(v) for k, v in interactions.items()}
            self.redis_client.setex(
                cache_key,
                30 * 24 * 3600,  # 30 days
                json.dumps(final_data)
            )
            logger.info(
                f"✓ Saved to Redis: {cache_key}, total types: {len(final_data)}, "
                f"total interactions: {sum(len(v) for v in final_data.values())}"
            )

            # Invalidate user's short-term vector cache
            vector_cache_key = f"user_vector:short_term:{event.account_id}"
            deleted = self.redis_client.delete(vector_cache_key)
            logger.info(f"✓ Invalidated vector cache: {vector_cache_key} (deleted={deleted})")

            return True

        except Exception as e:
            logger.error(f"Failed to update Redis cache: {e}", exc_info=True)
            return False

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
                        # Parse interaction event
                        event = InteractionEvent.from_redis_fields(fields)

                        logger.info(
                            f"User {event.account_id} → Job {event.job_id}: "
                            f"{event.event_type.value} (weight={event.get_weight()})"
                        )

                        # Task 1: Cache in Redis (same format as recommend.py)
                        cached = self._update_redis_cache(event)

                        # Log results
                        if cached:
                            logger.debug(f"{message_id}: cached successfully")
                        else:
                            logger.error(f"{message_id}: caching failed")

                        # ACK message
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id
                        )
                        processed_count += 1

                    except Exception as e:
                        logger.exception(
                            f"Error processing {message_id}: {e}")
                        # Still ACK to avoid infinite retries
                        self.redis_client.xack(
                            self.stream_name, self.consumer_group, message_id
                        )

            if processed_count > 0:
                logger.info(f"Processed {processed_count} interactions")

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
        logger.info("=" * 70)
        logger.info("Interaction Consumer Started")
        logger.info("=" * 70)
        logger.info(f"   Stream:    {self.stream_name}")
        logger.info(f"   Group:     {self.consumer_group}")
        logger.info("=" * 70)

        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                self.process_messages()
                retry_count = 0
            except KeyboardInterrupt:
                logger.info("Consumer interrupted by user")
                break
            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Consumer error (retry {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    logger.error("Max retries reached. Stopping.")
                    break
                import time
                time.sleep(5 * retry_count)

        logger.info("Interaction Consumer stopped")

    def stop(self):
        """Stop consumer"""
        logger.info("Stopping Interaction Consumer...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()