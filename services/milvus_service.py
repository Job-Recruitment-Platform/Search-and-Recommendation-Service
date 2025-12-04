"""Milvus service for vector database operations"""
import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.config import Config

logger = logging.getLogger(__name__)


jobs_collection_schema = {
    "name": "jobs",
    "description": "Jobs collection",
    "fields": [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=False),
        # Basic job info fields for search
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="skills", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="description",
                    dtype=DataType.VARCHAR, max_length=3000),
        # Filterable fields
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="job_role", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="seniority", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="min_experience_years", dtype=DataType.INT32),
        FieldSchema(name="work_mode", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="salary_min", dtype=DataType.INT32),
        FieldSchema(name="salary_max", dtype=DataType.INT32),
        FieldSchema(name="currency", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),

        # Other metadata fields
        FieldSchema(name="max_candidates", dtype=DataType.INT32),
        FieldSchema(name="date_posted", dtype=DataType.INT64),
        FieldSchema(name="date_expires", dtype=DataType.INT64),

        # Vector fields
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ],
    "indexes": [
        {
            "field_name": "dense_vector",
            "index_params": {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        },
        {
            "field_name": "sparse_vector",
            "index_params": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        },
    ],
}

users_collection_schema = {
    "name": "users",
    "description": "Users collection",
    "fields": [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=False),
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR, dim=1024),
    ],
    "indexes": [
        {
            "field_name": "dense_vector",
            "index_params": {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        },
    ],
}


class MilvusService:
    """Service for Milvus vector database operations"""

    def __init__(self):
        self.jobs_collection = None
        self.users_collection = None
        self.ef = None
        self.dense_dim = None
        self._setup()

    def _setup(self):
        """Setup Milvus connection and collection"""
        try:
            connections.connect(host=Config.MILVUS_HOST,
                                port=Config.MILVUS_PORT)
            logger.info(
                f"Connected to Milvus at {Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
            )

            # Initialize embedding function
            self.ef = BGEM3EmbeddingFunction(
                model_name=Config.EMBEDDING_MODEL_NAME,
                device=Config.EMBEDDING_DEVICE,
                use_fp16=Config.EMBEDDING_USE_FP16,
            )
            self.dense_dim = self.ef.dim["dense"]
            logger.info("Initialized BGEM3 embedding function")

            # Setup collection
            self.jobs_collection = self._setup_collection(
                jobs_collection_schema)
            self.users_collection = self._setup_collection(
                users_collection_schema)

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _setup_collection(self, schema: Dict[str, Any]):
        """Initialize Milvus collection"""
        collection_name = schema["name"]

        if not utility.has_collection(collection_name):
            fields = schema["fields"]
            collection_schema = CollectionSchema(
                fields=fields, description=schema.get("description", "")
            )
            collection = Collection(
                name=collection_name,
                schema=collection_schema
            )
            # Create indexes
            for index in schema.get("indexes", []):
                collection.create_index(
                    field_name=index["field_name"],
                    index_params=index["index_params"]
                )
            logger.info(f"Created new collection: {collection_name}")
        else:
            collection = Collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")

        collection.load()
        logger.info(f"Loaded collection: {collection_name}")
        return collection

    def generate_embeddings(self, texts: List[str]) -> Dict[str, object]:
        """Generate dense and sparse embeddings for given texts"""
        try:
            embeddings = self.ef.encode_documents(texts)

            # Handle sparse embeddings properly
            sparse_embeddings = []
            sparse_result = embeddings.get("sparse")

            # Check if sparse_result is a single matrix or list of matrices
            if hasattr(sparse_result, "tocoo"):
                # Single sparse matrix - convert to list
                sparse_matrices = [sparse_result]
            else:
                # Already a list
                sparse_matrices = sparse_result

            # Convert each sparse matrix to dict format for Milvus
            for sparse_matrix in sparse_matrices:
                if hasattr(sparse_matrix, "tocoo"):
                    # Convert scipy sparse matrix to dict {index: value}
                    coo = sparse_matrix.tocoo()
                    sparse_dict = {int(col): float(data)
                                   for col, data in zip(coo.col, coo.data)}
                    sparse_embeddings.append(sparse_dict)
                elif isinstance(sparse_matrix, dict):
                    # Already in dict format
                    sparse_embeddings.append(sparse_matrix)
                else:
                    logger.warning(
                        f"Unknown sparse format: {type(sparse_matrix)}")
                    sparse_embeddings.append({})

            return {
                "dense": embeddings["dense"],
                "sparse": sparse_embeddings,
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise

    def get_job_dense_vector(self, job_id: int) -> List[float]:
        """Fetch dense vector for a job by id. Returns empty list if not found."""
        try:
            results = self.jobs_collection.query(
                expr=f"id == {int(job_id)}",
                output_fields=["dense_vector"],
                limit=1,
            )
            if results and isinstance(results, list) and "dense_vector" in results[0]:
                return list(results[0]["dense_vector"])  # type: ignore
        except Exception as e:
            logger.warning(
                f"Failed to fetch dense vector for job {job_id}: {e}")
        return []

    def upsert_user_vector(self, user_id: int, dense_vector: List[float]) -> None:
        """Upsert a single user dense vector into the users collection."""
        try:
            # Delete existing if any
            try:
                self.users_collection.delete(expr=f"id in [{int(user_id)}]")
            except Exception:
                pass
            # Insert new entity
            entity = {"id": int(user_id), "dense_vector": dense_vector}
            self.users_collection.insert([entity])
            logger.info(f"Upserted user vector: user_id={user_id}")
        except Exception as e:
            logger.error(f"Failed to upsert user vector for {user_id}: {e}")
            raise

    def get_user_vector(self, user_id: int) -> Optional[List[float]]:
        """Get user vector from Milvus collection"""
        try:
            if not self.users_collection:
                return None
            results = self.users_collection.query(
                expr=f"id == {int(user_id)}",
                output_fields=["dense_vector"],
                limit=1,
            )
            if results and isinstance(results, list) and len(results) > 0:
                return list(results[0].get("dense_vector", []))
        except Exception as e:
            logger.warning(f"Failed to get user vector for {user_id}: {e}")
        return None

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
