"""Milvus service for vector database operations"""
import logging
from typing import List, Dict, Any
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    AnnSearchRequest,
    WeightedRanker,
    Function,
    FunctionType,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.config import Config

logger = logging.getLogger(__name__)


class MilvusService:
    """Service for Milvus vector database operations"""

    def __init__(self):
        self.jobs_collection = None
        self.user_collection = None
        self.ef = None
        self.dense_dim = 0
        self._setup()

    def _setup(self):
        """Setup Milvus connection and collection"""
        try:
            connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
            logger.info(
                f"Connected to Milvus at {Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
            )

            # Initialize embedding function
            self.ef = BGEM3EmbeddingFunction(
                model_name=Config.EMBEDDING_MODEL_NAME,
                device=Config.EMBEDDING_DEVICE,
                use_fp16=Config.EMBEDDING_USE_FP16,
                return_dense=True,
                return_sparse=True,
            )
            self.dense_dim = self.ef.dim["dense"]
            logger.info("Initialized BGEM3 embedding function")

            # Setup collection
            self._setup_jobs_collection()

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _setup_jobs_collection(self):
        """Initialize Milvus collection for jobs"""
        collection_name = "jobs"
        if not utility.has_collection(collection_name):

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),

                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="skills", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),

                # Filterable fields
                FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=100),  # e.g., "Acme Corp"
                FieldSchema(name="job_role", dtype=DataType.VARCHAR, max_length=100), # e.g., "Software Engineer"
                FieldSchema(name="seniority", dtype=DataType.VARCHAR, max_length=50), # e.g., "Junior", "Mid", "Senior", etc...
                FieldSchema(name="min_experience_years", dtype=DataType.INT32),
                FieldSchema(name="work_mode", dtype=DataType.VARCHAR, max_length=10), # e.g., "Remote", "Onsite", "Hybrid"
                FieldSchema(name="salary_min", dtype=DataType.INT32),                 # e.g., 60000
                FieldSchema(name="salary_max", dtype=DataType.INT32),                 # e.g., 120000
                FieldSchema(name="currency", dtype=DataType.VARCHAR, max_length=10),  # e.g., "USD", "EUR", "GBP", etc...
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),    # e.g., "PUBLISHED", "DRAFT", "PENDING", "EXPIRED", "CANCELED"
                
                # Other metadata fields
                FieldSchema(name="max_candidates", dtype=DataType.INT32),
                FieldSchema(name="date_posted", dtype=DataType.INT64),
                FieldSchema(name="date_expires", dtype=DataType.INT64),

                # Dense vector field
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]

            schema = CollectionSchema(
                fields=fields, 
                description="Jobs collection",
            )

            self.jobs_collection = Collection(
                name=collection_name,
                schema=schema
            )
           
            logger.info(f"Created new collection: {collection_name}")

            # Create index for dense vector
            self.jobs_collection.create_index(
                field_name="dense_vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )

            # Create index for sparse vector
            self.jobs_collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                },
            )

            logger.info(f"Created indexes for collection: {collection_name}")
        else:
            self.jobs_collection = Collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")

        self.jobs_collection.load()
        logger.info(f"Loaded collection: {collection_name}")
    
    def _setup_users_collection(self):
        """Initialize users collection"""
        
        collection_name = "users"
        
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50, is_primary=True, auto_id=False),
                
                # User metadata
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=255),
                
                # Skills from NLP
                FieldSchema(name="skills", dtype=DataType.ARRAY,element_type=DataType.VARCHAR,max_length=50,max_capacity=50),
                
                # Location from NLP
                FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
                
                # Previous companies (from Resume if can extract)
                FieldSchema(name="organizations", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=100, max_capacity=10, nullable=True),
                
                # Vectors
                FieldSchema(
                    name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dense_dim
                ),
                FieldSchema(
                    name="sparse_vector",
                    dtype=DataType.SPARSE_FLOAT_VECTOR
                ),
                
                # Timestamps
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
            ]
            
            schema = CollectionSchema(fields=fields, description="Users collection")
            self.user_collection = Collection(
                name=collection_name,
                schema=schema
            )
            
            # Create indexes
            self.user_collection.create_index(
                field_name="dense_vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )
            
            self.user_collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP"
                }
            )
        
            logger.info(f"Created new collection: {collection_name}")
        else:
            self.user_collection = Collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        self.user_collection.load()
        logger.info(f"Loaded collection: {collection_name}")
        return self.user_collection
    
    
    
    def generate_embeddings(self, texts: List[str]) -> Dict[str, object]:
        """Generate dense and sparse embeddings for given texts"""
        try:
            embeddings = self.ef.encode_documents(texts)

            return {
                "dense": embeddings["dense"],
                "sparse": embeddings["sparse"],
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def upsert_jobs(self, entities: List[Dict[str, Any]]) -> int:
        """Upsert jobs into Milvus collection"""
        try:
            if not entities:
                raise ValueError("No jobs to upsert")
            
            self.delete_jobs([entity["id"] for entity in entities])
            
            self.jobs_collection.insert(entities)
            logger.info(f"Inserted {len(entities)} jobs into Milvus")
            return len(entities)
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    def delete_jobs(self, job_ids: List[int]) -> int:
        """Delete jobs from Milvus collection"""
        try:
            if not job_ids:
                return 0
            id_expr = ",".join(str(i) for i in job_ids)
            res = self.jobs_collection.delete(expr=f"id in [{id_expr}]")
            deleted = res.delete_count if hasattr(res, "delete_count") else 0
            logger.info(f"Deleted {deleted} jobs from Milvus")
            return deleted
        except Exception as e:
            logger.warning(f"Delete failed: {e}")
            return 0

