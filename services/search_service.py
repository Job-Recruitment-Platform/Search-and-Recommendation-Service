"""Search service for job search operations"""
import logging
import time
from typing import List, Tuple
from services.milvus_service import MilvusService
from app.config import Config
from models.search import SearchWeights
from models.embeddings import Embeddings
from models.pagination import PaginationInfo
from pymilvus import AnnSearchRequest, WeightedRanker

logger = logging.getLogger(__name__)


class SearchService:
    """Service for job search operations"""

    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        threshold: float = Config.SEARCH_THRESHOLD,
    ) -> Tuple[List[int], PaginationInfo]:
        """Perform hybrid search for jobs
        Returns:
            Tuple of (job_ids, pagination_info)
        """

        query_embedding = self.milvus_service.generate_embeddings([query])
        dense_vec = query_embedding.get("dense")[0]
        sparse_vec = query_embedding.get("sparse")[0]

        # Normalize sparse row to dict {index: value} if needed
        if hasattr(sparse_vec, "tocoo"):
            coo = sparse_vec.tocoo()
            sparse_vec = {int(j): float(v) for j, v in zip(coo.col, coo.data)}

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit+1
        )
        
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit+1
        )
        results = self.milvus_service.jobs_collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(float(0.5), float(0.5)),
            offset=offset,
            limit=limit+1,
            output_fields=["id"],
        )
        job_ids = []
        for hits in results:
            for hit in hits:
                if hit.score is None or hit.score < float(threshold):
                    continue
                job_ids.append(hit.entity.get("id"))
                logger.info(
                    f"Hit: id={hit.entity.get('id')}, score={hit.score}, threshold={threshold}"
                )
                
        has_next = len(job_ids) > limit
        job_ids = job_ids[:limit]
        # Build pagination info
        pagination = PaginationInfo(
            limit=limit,
            offset=offset,
            count=len(job_ids),
            has_next=has_next,
            has_prev=offset > 0,
        )

        return job_ids, pagination

