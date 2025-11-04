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
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_STREAM_NAME: str = os.getenv("REDIS_STREAM_NAME", "outbox-events")
    REDIS_CONSUMER_GROUP: str = os.getenv("REDIS_CONSUMER_GROUP", "milvus-sync")
    REDIS_CONSUMER_NAME: str = os.getenv("REDIS_CONSUMER_NAME", "worker-1")
    
    # Embedding model configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_USE_FP16: bool = os.getenv("EMBEDDING_USE_FP16", "false").lower() == "true"
    
    # Search configuration
    SEARCH_DEFAULT_LIMIT: int = int(os.getenv("SEARCH_DEFAULT_LIMIT", "10"))
    SEARCH_DEFAULT_OFFSET: int = int(os.getenv("SEARCH_DEFAULT_OFFSET", "0"))
    SEARCH_THRESHOLD: float = float(os.getenv("SEARCH_THRESHOLD", "0.5"))
