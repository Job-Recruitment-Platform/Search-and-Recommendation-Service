"""Application configuration"""
import os


class Config:
    """Application configuration loaded from environment variables"""

    # Flask configuration
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "8000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Milvus configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")

    # API python
    API_SERVER_HOST = os.getenv("API_SERVER_HOST", "localhost")

    # Internal token
    INTERNAL_API_TOKEN: str = os.getenv(
        "INTERNAL_API_TOKEN", "68527e7f-4c0c-4c02-8df2-9c5b639cee77")

    # Postgres
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "root")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "niggamove")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "jrp")

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Stream 1: Outbox Events (Job sync to Milvus)
    # Consumed by: OutboxEventConsumer
    # Message format: {id, aggregateType, aggregateId, eventType, payload, occurredAt, traceId, attempts}
    OUTBOX_STREAM_NAME: str = "outbox-events"
    OUTBOX_CONSUMER_GROUP: str = "outbox-processor-group"
    OUTBOX_CONSUMER_NAME: str = "python-sync-worker-1"

    # Stream 2: User Interactions (Recommendation signals)
    # Consumed by: InteractionConsumer
    # Message format: {accountId, jobId, eventType, metadata, occurredAt}
    INTERACTION_STREAM_NAME: str = "user-interactions"
    INTERACTION_CONSUMER_GROUP: str = "recommend-service-group"
    INTERACTION_CONSUMER_NAME: str = "python-recommend-worker-1"

    # Embedding model configuration
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_USE_FP16: bool = os.getenv(
        "EMBEDDING_USE_FP16", "false").lower() == "true"

    # Search configuration
    SEARCH_DEFAULT_LIMIT: int = int(os.getenv("SEARCH_DEFAULT_LIMIT", "10"))
    SEARCH_DEFAULT_OFFSET: int = int(os.getenv("SEARCH_DEFAULT_OFFSET", "0"))
    SEARCH_THRESHOLD: float = float(os.getenv("SEARCH_THRESHOLD", "0.3"))

    # Recommendation configuration
    CANDIDATE_API_BASE_URL: str = os.getenv(
        "CANDIDATE_API_BASE_URL", "http://localhost:8080")
    INTERACTION_STREAM_NAME: str = os.getenv(
        "INTERACTION_STREAM_NAME", "outbox-events")
    INTERACTION_CONSUMER_GROUP: str = os.getenv(
        "INTERACTION_CONSUMER_GROUP", "recommend-service-group")
    INTERACTION_HALF_LIFE_DAYS: float = float(
        os.getenv("INTERACTION_HALF_LIFE_DAYS", "30"))

    # CF model configuration
    CF_MODEL_PATH: str = os.getenv(
        "CF_MODEL_PATH", "CFModel/models/cf_model.pkl")
    TRAINING_SCHEDULE_TIME: str = os.getenv("TRAINING_SCHEDULE_TIME", "02:00")


INTERACTION_WEIGHTS = {
    # Positive signals
    'APPLY': 1.0,  # Strongest positive signal
    "SAVE": 0.6,  # Medium weight for saving jobs
    # "CLICK" == "VIEW"
    "CLICK_FROM_SIMILAR": 0.2,  # Clicks from similar jobs section
    "CLICK_FROM_RECOMMENDED": 0.25,  # Clicks from recommended jobs section
    "CLICK_FROM_SEARCH": 0.4,  # Clicks from search results
    # Negative signals
    "SKIP_FROM_SIMILAR": -0.05,  # Skip from similar jobs section
    "SKIP_FROM_RECOMMENDED": -0.15,  # Skip from recommended jobs section
    "SKIP_FROM_SEARCH": -0.1,  # Skip from search
}
