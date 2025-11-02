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
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.config import Config

logger = logging.getLogger(__name__)


class MilvusService:
    """Service for Milvus vector database operations"""

    def __init__(self):
        self.jobs_collection = None
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
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                ),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(
                    name="skills",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=50,
                    max_capacity=50,
                ),
                FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="job_role", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="seniority", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="min_experience_years", dtype=DataType.INT32),
                FieldSchema(name="work_mode", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="salary_min", dtype=DataType.INT32),
                FieldSchema(name="salary_max", dtype=DataType.INT32),
                FieldSchema(name="currency", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="max_candidates", dtype=DataType.INT32),
                FieldSchema(name="date_posted", dtype=DataType.INT64),
                FieldSchema(name="date_expires", dtype=DataType.INT64),
                FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(
                    name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dense_dim,
                ),
            ]
            self.jobs_collection = Collection(
                name=collection_name, schema=CollectionSchema(fields)
            )
            logger.info(f"Created new collection: {collection_name}")

            # Create index for dense vector
            self.jobs_collection.create_index(
                field_name="dense_vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 24, "efConstruction": 200},
                },
            )

            # Create index for sparse vector (BM25)
            self.jobs_collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {
                        "inverted_index_algo": "DAAT_WAND",
                        "drop_ratio_build": 0.2,
                    },
                },
            )
            logger.info(f"Created indexes for collection: {collection_name}")
        else:
            self.jobs_collection = Collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")

        self.jobs_collection.load()
        logger.info(f"Loaded collection: {collection_name}")

    def generate_embeddings(self, texts: List[str]) -> Dict[str, List]:
        """Generate embeddings: returns dict with 'dense' and 'sparse'."""
        try:
            return self.ef(texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def insert_jobs(self, entities: List) -> int:
        """Insert jobs into Milvus collection"""
        try:
            mr = self.jobs_collection.insert(entities)
            inserted = mr.insert_count if hasattr(mr, "insert_count") else len(entities[0])
            logger.info(f"Inserted {inserted} jobs into Milvus")
            return inserted
        except Exception as e:
            logger.error(f"Failed to insert jobs: {e}")
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

    @staticmethod
    def _convert_sparse_vector(sparse_vec) -> Dict[int, float]:
        """
        Convert sparse vector to dict format {index: value}.
        Handles scipy sparse matrices, lists, and other formats.
        """
        if isinstance(sparse_vec, dict):
            return sparse_vec
        
        # Handle scipy sparse matrix formats (CSR, CSC, COO, etc.)
        if hasattr(sparse_vec, 'tocoo'):
            try:
                coo = sparse_vec.tocoo()
                # For 1D sparse vector, col indices are the positions
                return {int(i): float(v) for i, v in zip(coo.col, coo.data)}
            except Exception as e:
                logger.warning(f"Failed to convert sparse matrix: {e}, using empty dict")
                return {}
        
        # Handle list, numpy array, or other iterable
        if hasattr(sparse_vec, '__iter__') and not isinstance(sparse_vec, str):
            try:
                return {i: float(v) for i, v in enumerate(sparse_vec) if v != 0}
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not convert sparse vector: {e}, type: {type(sparse_vec)}")
                return {}
        
        logger.warning(f"Unknown sparse vector format, type: {type(sparse_vec)}, using empty dict")
        return {}

    def hybrid_search(
        self,
        dense_vec,
        sparse_vec,
        dense_weight=1.0,
        sparse_weight=1.0,
        offset=0,
        limit=10,
    ):
        """Hybrid search in jobs collection"""
        # Convert sparse vector to dict format required by Milvus
        sparse_vec = self._convert_sparse_vector(sparse_vec)
        
        dense_req = AnnSearchRequest(
            [dense_vec], "dense_vector", {"metric_type": "COSINE"}, limit=limit
        )
        sparse_req = AnnSearchRequest(
            [sparse_vec], "sparse_vector", {"metric_type": "IP"}, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)

        # Only need id field for search results
        output_fields = ["id"]

        return self.jobs_collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            offset=offset,
            output_fields=output_fields,
        )[0]

