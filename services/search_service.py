"""Search service for job search operations"""
import logging
import time
from typing import List, Tuple
from services.milvus_service import MilvusService
from app.config import Config
from models.search import SearchWeights
from models.embeddings import Embeddings
from models.pagination import PaginationInfo

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
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> Tuple[List[int], PaginationInfo]:
        """Perform hybrid search for jobs
        
        Returns:
            Tuple of (job_ids, pagination_info)
        """
        start = time.time()

        try:
            embeddings_dict = self.milvus_service.generate_embeddings([query])
            embeddings = Embeddings.from_dict(embeddings_dict)
            logger.info(f"Generated embeddings in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

        # Request more results to check if there are more pages
        # We'll request limit + 1 to determine has_next
        results = self.milvus_service.hybrid_search(
            embeddings.get_dense_vector(0),
            embeddings.get_sparse_vector(0),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            limit=limit + 1,  # Request one extra to check has_next
            offset=offset,
        )

        logger.info(
            f"Search completed in {time.time() - start:.2f}s, found {len(results)} results"
        )

        job_ids = []
        
        # Filter results by threshold and extract only IDs
        for hit in results:
            score = float(hit.score)
            if score < Config.SEARCH_SCORE_THRESHOLD:
                continue
            
            job_id = int(hit.get("id"))
            if job_id:
                job_ids.append(job_id)

        # Determine has_next: if we got limit+1 results, there might be more
        has_next = len(job_ids) > limit
        
        # Return only up to limit results
        if has_next:
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

