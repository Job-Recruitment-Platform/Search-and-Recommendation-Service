from typing import Dict, Any, List, Optional
import math
import json
from datetime import datetime, timezone
from collections import Counter, defaultdict
from threading import Lock
from pathlib import Path
import logging
import redis
import numpy as np
import os
import pickle
import requests
import threading
import time

from services.milvus_service import MilvusService
from app.config import INTERACTION_WEIGHTS, Config

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service
        # Try load CF model (optional)
        self.cf_model = None
        self.cf_user_id_to_index = None
        self.cf_item_id_to_index = None
        self.cf_index_to_item_id = None
        self.cf_index_to_user_id = None
        self.model_load_lock = Lock()
        self.last_model_mtime = None
        self._load_cf_model()
        # Initialize Redis client for short-term vector caching
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=False  # We'll handle JSON encoding/decoding manually
            )
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            print(f"Warning: Failed to connect to Redis: {e}")
            self.redis_client = None

        # Start interaction stream consumer in background
        self._start_interaction_consumer()

    def recommend(
        self,
        user_id: int,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Hybrid recommendation entrypoint.

        Tách candidate generation và ranking:
        - Candidate: kết hợp CF + content để tăng coverage
        - Ranking: áp dụng trọng số nguồn + exploration post-processing
        """
        try:
            user_profile = self._get_user_profile(user_id)
            user_interactions = self._get_user_interactions(user_id)

            candidates = self._generate_candidates(
                user_id=user_id,
                user_profile=user_profile,
                user_interactions=user_interactions,
                top_k=top_k,
            )

            if not candidates:
                return self._get_popular_jobs(top_k)

            ranked = self._rank_candidates(
                user_id=user_id,
                candidates=candidates,
                user_interactions=user_interactions,
                top_k=top_k,
            )

            return ranked or self._get_popular_jobs(top_k)

        except Exception as e:
            print(f"Error in recommend(): {e}")
            return self._get_popular_jobs(top_k)

    def _generate_candidates(
        self,
        user_id: int,
        user_profile: Optional[Dict[str, Any]],
        user_interactions: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Hybrid candidate generation (CF + content + tiêu chuẩn fallback)."""

        candidate_map: Dict[int, Dict[str, Any]] = {}

        def ensure_candidate(job_id: int) -> Dict[str, Any]:
            entry = candidate_map.get(job_id)
            if entry is None:
                entry = {
                    "job_id": job_id,
                    "score_sources": {},
                    "sources": set(),
                }
                candidate_map[job_id] = entry
            return entry

        # CF candidates
        for cand in self._generate_cf_candidates(user_id=user_id, limit=top_k * 2):
            entry = ensure_candidate(cand["job_id"])
            entry["score_sources"]["cf"] = cand["score"]
            entry["sources"].add("cf")

        # Content-based candidates
        for cand in self._generate_content_candidates(
            user_id=user_id,
            user_profile=user_profile,
            user_interactions=user_interactions,
            limit=top_k * 3,
        ):
            entry = ensure_candidate(cand["job_id"])
            entry["score_sources"]["content"] = cand["score"]
            entry["sources"].add("content")

        # Coverage boost: nếu vẫn thiếu candidate, bổ sung popular
        if not candidate_map or len(candidate_map) < top_k:
            for cand in self._generate_popular_candidates(limit=top_k * 2):
                entry = ensure_candidate(cand["job_id"])
                entry["score_sources"].setdefault("popular", cand["score"])
                entry["sources"].add("popular")

        return list(candidate_map.values())

    def _load_cf_model(self, force_reload: bool = False) -> bool:
        """Load CF model with hot reload support

        Args:
            force_reload: Force reload even if file hasn't changed

        Returns:
            True if model was loaded/reloaded, False otherwise
        """
        try:
            path = getattr(Config, "CF_MODEL_PATH", "")
            if not path:
                logger.warning("CF_MODEL_PATH not configured")
                return False

            model_path = Path(path)
            if not model_path.exists():
                logger.warning(f"CF model not found at {model_path}")
                return False

            # Check if model file changed
            current_mtime = model_path.stat().st_mtime

            if not force_reload and self.last_model_mtime == current_mtime:
                # Model unchanged
                return False

            # Load model with thread safety
            with self.model_load_lock:
                logger.info(f"Loading CF model from {model_path}...")

                with open(model_path, "rb") as f:
                    data = pickle.load(f)

                self.cf_model = data.get("model")
                self.cf_user_id_to_index = data.get("user_id_to_index", {})
                self.cf_item_id_to_index = data.get("item_id_to_index", {})
                self.cf_index_to_item_id = data.get("index_to_item_id", {})
                self.cf_index_to_user_id = data.get("index_to_user_id", {})

                self.last_model_mtime = current_mtime

                logger.info(
                    f"✓ CF model loaded: {len(self.cf_user_id_to_index)} users, "
                    f"{len(self.cf_item_id_to_index)} jobs"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to load CF model: {e}")
            return False

    def _generate_cf_candidates(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Sinh candidate từ mô hình CF (ALS); trả [] nếu không khả dụng."""
        try:
            if not self.cf_model or not self.cf_user_id_to_index or not self.cf_index_to_item_id:
                return []
            if user_id not in self.cf_user_id_to_index:
                return []
            user_idx = self.cf_user_id_to_index[user_id]
            # Build dummy user-items row from available data: need the user_item row. We don't have matrix here.
            # Use model.recommend with user_items=None not allowed; instead pass empty csr row.
            from scipy.sparse import csr_matrix
            max_item_index = max(self.cf_index_to_item_id.keys()
                                 ) if self.cf_index_to_item_id else -1
            num_items = max_item_index + 1 if max_item_index >= 0 else 1
            user_items = csr_matrix((1, num_items))
            ids, scores = self.cf_model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=limit,
                filter_already_liked_items=True,
            )
            results: List[Dict[str, Any]] = []
            for idx, score in zip(ids, scores):
                job_id = self.cf_index_to_item_id.get(int(idx))
                if job_id is None:
                    continue
                results.append({
                    "job_id": int(job_id),
                    "score": float(score),
                })
            return results
        except Exception:
            return []

    def _generate_popular_candidates(
        self,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Chuyển danh sách job phổ biến thành candidate chuẩn hoá."""
        popular_jobs = self._get_popular_jobs(limit)
        candidates: List[Dict[str, Any]] = []
        for rank, job in enumerate(popular_jobs or []):
            job_id = job.get("job_id")
            if job_id is None:
                continue
            score = job.get("score")
            if score is None:
                score = max(0.0, (limit - rank) / max(1, limit))
            candidate = {
                "job_id": int(job_id),
                "score": float(score),
            }
            candidates.append(candidate)
        return candidates

    def _rank_candidates(
        self,
        user_id: int,
        candidates: List[Dict[str, Any]],
        user_interactions: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Xếp hạng candidate với chiến lược hybrid + exploration."""

        seen_job_ids = self._collect_seen_job_ids(user_interactions)
        ranked_results: List[Dict[str, Any]] = []

        for candidate in candidates:
            job_id = candidate["job_id"]
            score_sources = candidate.get("score_sources", {})

            cf_score = score_sources.get("cf")
            content_score = score_sources.get("content")
            popular_score = score_sources.get("popular")

            combined = 0.0
            if cf_score is not None:
                combined += 0.65 * self._normalize_cf_score(cf_score)
            if content_score is not None:
                combined += 0.35 * self._normalize_content_score(content_score)
            if popular_score is not None and cf_score is None and content_score is None:
                combined += 0.25 * self._normalize_content_score(popular_score)

            # Boost nếu candidate đến từ nhiều nguồn
            num_sources = len(candidate.get("sources", []))
            if num_sources > 1:
                combined += 0.05 * (num_sources - 1)

            # Exploration bonus cho job chưa từng tương tác
            if job_id not in seen_job_ids:
                combined += self._exploration_bonus(job_id)
            else:
                combined *= 0.85  # giảm ưu tiên job đã xem

            ranked_results.append({
                "job_id": job_id,
                "score": combined,
            })

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        top_results = ranked_results[:top_k]
        return [{"job_id": item["job_id"]} for item in top_results]

    def _normalize_cf_score(self, score: float) -> float:
        """Chuyển score CF (ALS) về [0, 1] bằng logistic."""
        try:
            return 1.0 / (1.0 + math.exp(-float(score)))
        except Exception:
            return 0.0

    def _normalize_content_score(self, score: float) -> float:
        """Chuẩn hoá score cosine về [0, 1]."""
        try:
            val = float(score)
        except Exception:
            return 0.0
        val = max(-1.0, min(1.0, val))
        return (val + 1.0) / 2.0

    def _exploration_bonus(self, job_id: int) -> float:
        """Bonus deterministic cho exploration."""
        hashed = hash(("explore", int(job_id)))
        noise = (hashed & 0xFFFF) / 0xFFFF  # 0..1
        return 0.03 + 0.02 * noise

    def _collect_seen_job_ids(self, interactions: Optional[Dict[str, Any]]) -> set:
        """Tập job user đã tương tác."""
        seen: set = set()
        if not interactions:
            return seen
        for entries in interactions.values():
            if isinstance(entries, dict):
                seen.update(int(jid)
                            for jid in entries.keys() if str(jid).isdigit())
            elif isinstance(entries, (list, tuple, set)):
                for jid in entries:
                    try:
                        seen.add(int(jid))
                    except Exception:
                        continue
        return seen

    def _hydrate_candidate_metadata(self, candidate_map: Dict[int, Dict[str, Any]]) -> None:
        """Bổ sung metadata còn thiếu cho candidate (nếu có API hỗ trợ)."""
        missing_ids = [job_id for job_id,
                       data in candidate_map.items() if not data.get("metadata")]
        if not missing_ids:
            return
        try:
            metadata_list = self._get_jobs_metadata(
                [str(job_id) for job_id in missing_ids])
        except Exception:
            return
        if not metadata_list:
            return
        metadata_by_id = {}
        for item in metadata_list:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("job_id") or item.get("id")
            if raw_id is None:
                continue
            try:
                metadata_by_id[int(raw_id)] = item
            except Exception:
                continue
        for job_id, meta in metadata_by_id.items():
            if job_id not in candidate_map:
                continue
            entry = candidate_map[job_id]
            metadata = entry.setdefault("metadata", {})
            metadata.setdefault("title", meta.get("title", ""))
            metadata.setdefault("company", meta.get("company", ""))
            metadata.setdefault("location", meta.get("location", ""))
            metadata.setdefault("salary_range", meta.get("salary_range", ""))

    def _generate_content_candidates(
        self,
        user_id: int,
        user_profile: Optional[Dict[str, Any]],
        user_interactions: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Sinh candidate dựa trên vector nội dung (Milvus)."""

        if not user_profile and not user_interactions:
            return []

        user_vector: List[float] = []
        if user_profile:
            user_vector = self._calculate_user_vector(
                user_profile, user_interactions or {})
        elif user_interactions:
            user_vector = self._calculate_short_term_user_vector(
                user_id, user_interactions)

        if not user_vector:
            return []

        try:
            # Use Milvus collection hybrid_search with dense vector only
            from pymilvus import AnnSearchRequest

            dense_req = AnnSearchRequest(
                data=[user_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=max(1, limit),
            )

            # hybrid_search requires rerank parameter, use WeightedRanker with weight 1.0 for dense only
            from pymilvus import WeightedRanker

            search_results = self.milvus_service.jobs_collection.hybrid_search(
                reqs=[dense_req],
                rerank=WeightedRanker(float(1.0)),
                limit=max(1, limit),
                output_fields=["id"],
            )
        except Exception as err:
            print(f"Warning: content candidate search failed: {err}")
            return []

        if not search_results or len(search_results) == 0 or not search_results[0]:
            return []

        hits = search_results[0]
        candidates: List[Dict[str, Any]] = []
        for hit in hits:
            # Handle Milvus hit object format (similar to SearchService)
            job_id = None
            score = 0.0

            if hasattr(hit, "entity"):
                # Milvus hit object
                job_id = hit.entity.get("id") if hasattr(
                    hit.entity, "get") else getattr(hit.entity, "id", None)
                score = getattr(hit, "score", 0.0)
            elif isinstance(hit, dict):
                # Dictionary format
                job_id = hit.get("id") or hit.get("job_id")
                score = hit.get("score", 0.0)
            else:
                # Try to get attributes directly
                job_id = getattr(hit, "id", None)
                score = getattr(hit, "score", 0.0)

            if job_id is None:
                continue

            candidate = {
                "job_id": int(job_id),
                "score": float(score) if score is not None else 0.0,
            }
            candidates.append(candidate)

        return candidates

    def _calculate_long_term_user_vector(
        self,
        user_profile: Dict[str, Any]
    ) -> List[float]:
        """Calculate long-term user vector from profile data only.

        This represents the user's stable preferences based on their profile
        (skills, education, location, preferences). This vector is saved to Milvus
        as it represents long-term user characteristics.

        Args:
            user_profile: User profile dictionary with skills, education, location, etc.

        Returns:
            Normalized dense vector representing long-term user preferences
        """
        # Build profile text (without interaction insights for long-term vector)
        profile_text = self._build_profile_text(
            user_profile, user_interactions=None)
        profile_dense = self._embed_text_to_dense(profile_text)

        if not profile_dense:
            return []

        if isinstance(profile_dense, list) and len(profile_dense) > 0:
            if isinstance(profile_dense[0], list):
                profile_dense = profile_dense[0]
            elif isinstance(profile_dense[0], np.ndarray):
                profile_dense = profile_dense[0].tolist()

        # Convert to numpy array để normalize
        profile_dense = np.array(profile_dense, dtype=np.float32)
        if len(profile_dense.shape) > 1:
            # Nếu vẫn là 2D, flatten
            profile_dense = profile_dense.flatten()

        # Normalize to unit vector using numpy
        profile_dense = self._normalize_vector(profile_dense.tolist())

        # Save to Milvus (long-term storage)
        user_id = user_profile.get("id")
        if user_id is not None:
            try:
                self.milvus_service.upsert_user_vector(
                    int(user_id), profile_dense)
            except Exception as e:
                print(
                    f"Warning: Failed to upsert long-term user vector for user {user_id}: {e}")

        return profile_dense

    def _calculate_short_term_user_vector(
        self,
        user_id: int,
        user_interactions: Dict[str, Any],
        cache_ttl: int = 3600
    ) -> List[float]:
        """Calculate short-term user vector from recent interactions.

        This represents the user's current interests based on their recent behavior
        (clicks, saves, applies). This vector is cached in Redis as it changes
        frequently and needs to be updated in real-time.

        Args:
            user_id: User ID
            user_interactions: Dictionary of user interactions with timestamps
            cache_ttl: Time-to-live for Redis cache in seconds (default: 1 hour)

        Returns:
            Normalized dense vector representing short-term user interests
        """
        # Check Redis cache first
        cache_key = f"user_vector:short_term:{user_id}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    if isinstance(cached, (bytes, bytearray)):
                        cached = cached.decode()
                    return json.loads(cached)
            except Exception as e:
                print(f"Warning: Failed to read from Redis cache: {e}")

        # Compute behavior vector from interactions
        behavior_dense = self._compute_behavior_dense(
            user_interactions,
            self.milvus_service.dense_dim
        )

        if not behavior_dense:
            return []

        # Normalize to unit vector using numpy
        behavior_dense = self._normalize_vector(behavior_dense)

        # Cache in Redis (short-term storage)
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(behavior_dense)
                )
            except Exception as e:
                print(
                    f"Warning: Failed to cache short-term user vector in Redis: {e}")

        return behavior_dense

    def invalidate_short_term_cache(self, user_id: int) -> None:
        """Invalidate the short-term user vector cache in Redis.

        Call this method when new user interactions are recorded to ensure
        the short-term vector is recalculated on the next request.

        Args:
            user_id: User ID whose cache should be invalidated
        """
        if self.redis_client:
            try:
                cache_key = f"user_vector:short_term:{user_id}"
                self.redis_client.delete(cache_key)
            except Exception as e:
                print(
                    f"Warning: Failed to invalidate short-term cache for user {user_id}: {e}")

    def _calculate_user_vector(
        self,
        user_profile: Dict[str, Any],
        user_interactions: Dict[str, Any]
    ) -> List[float]:
        """Calculate combined user dense vector from profile and interactions.

        This method combines long-term (profile) and short-term (interactions) vectors
        using adaptive weighting based on interaction count:
        - Cold start (<5): 90% profile, 10% behavior
        - Growing (5-20): 60% profile, 40% behavior  
        - Mature (>20): 30% profile, 70% behavior

        Note: This method uses the separate long-term and short-term calculation methods
        internally. For better performance, consider using those methods directly.
        """
        # Get long-term vector (from profile, saved in Milvus)
        long_term_vector = self._calculate_long_term_user_vector(user_profile)

        # Get short-term vector (from interactions, cached in Redis)
        user_id = user_profile.get("id")
        if user_id is None:
            return long_term_vector

        short_term_vector = self._calculate_short_term_user_vector(
            int(user_id),
            user_interactions
        )

        # If no interactions, return long-term vector only
        if not short_term_vector:
            return long_term_vector

        # Count interactions for adaptive weights
        interaction_count = self._count_total_interactions(user_interactions)

        # Adaptive weights
        if interaction_count < 5:
            alpha, beta = 0.9, 0.1  # Cold start
        elif interaction_count < 20:
            alpha, beta = 0.6, 0.4  # Growing
        else:
            alpha, beta = 0.3, 0.7  # Mature

        # Combine long-term and short-term vectors
        final_dense = self._combine_vectors(
            long_term_vector, short_term_vector, alpha, beta)

        # Normalize to unit vector using numpy
        final_dense = self._normalize_vector(final_dense)

        return final_dense

    def _count_total_interactions(self, interactions: Dict[str, Any]) -> int:
        """Count total number of interactions across all types"""
        count = 0
        if not isinstance(interactions, dict):
            return 0

        for entries in interactions.values():
            if isinstance(entries, dict):
                count += len(entries)
            elif isinstance(entries, (list, tuple, set)):
                count += len(entries)
        return count

    def _to_text(self, value: Any) -> str:
        """Convert any value to text representation"""
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value if v is not None)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value)

    def _build_profile_text(
        self,
        user_profile: Dict[str, Any],
        user_interactions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build text representation of user profile"""
        parts: List[str] = []

        # Skills
        skills = user_profile.get("skills")
        if skills:
            parts.append(f"Skills: {self._to_text(skills)}")

        # Education
        education = user_profile.get("education")
        if education:
            parts.append(f"Education: {self._to_text(education)}")

        # Location
        location = user_profile.get("location")
        if location:
            parts.append(f"Location: {self._to_text(location)}")

        # Preferences
        preferences = user_profile.get("preferences") or {}
        if isinstance(preferences, dict):
            if "remote" in preferences:
                parts.append(
                    f"Prefers remote: {self._to_text(preferences.get('remote'))}")
            if "relocation" in preferences:
                parts.append(
                    f"Open to relocation: {self._to_text(preferences.get('relocation'))}")

        # Add interaction insights if available
        if user_interactions:
            insights = self._extract_interaction_insights(user_interactions)
            if insights.get('preferred_skills'):
                parts.append(
                    f"Interested in: {', '.join(insights['preferred_skills'][:5])}")

        text = "\n".join(parts).strip()
        return text or "No profile data"

    def _extract_interaction_insights(self, interactions: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract insights from positive interactions"""
        # Collect positive interaction job IDs
        positive_job_ids = []
        positive_types = {"APPLY", "SAVE", "CLICK_FROM_SEARCH"}

        for key, entries in interactions.items():
            if str(key).upper() not in positive_types:
                continue

            if isinstance(entries, dict):
                positive_job_ids.extend(str(jid) for jid in entries.keys())
            elif isinstance(entries, (list, tuple, set)):
                positive_job_ids.extend(str(jid) for jid in entries)

        if not positive_job_ids:
            return {}

        # Get job metadata
        try:
            jobs_metadata = self._get_jobs_metadata(positive_job_ids)
        except Exception:
            return {}

        # Extract patterns
        skills = []
        for job in jobs_metadata:
            if job.get('required_skills'):
                skills.extend(job['required_skills'])

        return {
            'preferred_skills': [s for s, _ in Counter(skills).most_common(10)]
        }

    def _get_jobs_metadata(self, job_ids: List[str]) -> List[Dict]:
        """Get job metadata - implement based on your architecture"""
        # TODO: Query from database or job service
        # Should return list of dicts with 'required_skills', 'industry', etc.
        return []

    def _embed_text_to_dense(self, text: str) -> List[float]:
        """Embed text to dense vector using BGE-M3"""
        embeddings = self.milvus_service.generate_embeddings([text])
        if not embeddings or "dense" not in embeddings:
            return []

        dense = embeddings["dense"]
        # generate_embeddings trả về list of lists, cần lấy phần tử đầu tiên
        if isinstance(dense, list) and len(dense) > 0:
            # Kiểm tra xem có phải nested list không
            if isinstance(dense[0], list):
                return list(dense[0])  # Lấy vector đầu tiên
            elif isinstance(dense[0], np.ndarray):
                return dense[0].tolist()
            else:
                return list(dense)  # Nếu đã là flat list
        return []

    def _compute_behavior_dense(
        self,
        interactions: Dict[str, Any],
        dimension: int
    ) -> List[float]:
        """Compute behavior vector from interactions with time decay"""

        allowed_keys = {
            "APPLY", "SAVE", "CLICK",
            "CLICK_FROM_SIMILAR", "CLICK_FROM_RECOMMENDED", "CLICK_FROM_SEARCH",
            "SKIP_FROM_SIMILAR", "SKIP_FROM_RECOMMENDED", "SKIP_FROM_SEARCH",
        }

        half_life_days = getattr(Config, "INTERACTION_HALF_LIFE_DAYS", 30)
        now_ts = datetime.now(timezone.utc).timestamp()

        # Use numpy array for efficient accumulation
        acc = np.zeros(dimension, dtype=np.float32)
        weight_sum: float = 0.0

        if not isinstance(interactions, dict):
            return [0.0] * dimension

        for raw_key, entries in interactions.items():
            key_upper = str(raw_key).upper()
            if key_upper not in allowed_keys or key_upper not in INTERACTION_WEIGHTS:
                continue

            base_w = float(INTERACTION_WEIGHTS[key_upper])

            if isinstance(entries, dict):
                items = entries.items()
            elif isinstance(entries, (list, tuple, set)):
                items = [(jid, None) for jid in entries]
            else:
                continue

            for job_id, ts in items:
                try:
                    j_id = int(job_id)
                except Exception:
                    continue

                job_vec = self.milvus_service.get_job_dense_vector(j_id)
                if not job_vec or len(job_vec) != dimension:
                    continue

                decay = self._exp_time_decay(ts, now_ts, half_life_days)
                w = base_w * decay

                # Use numpy for vector addition
                acc += w * np.array(job_vec, dtype=np.float32)
                weight_sum += abs(w)

        # Normalize using numpy
        if weight_sum > 1e-8:
            acc = acc / weight_sum
            return acc.tolist()

        return [0.0] * dimension

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length using numpy.

        Args:
            vector: Input vector as list of floats

        Returns:
            Normalized vector as list of floats
        """
        if not vector:
            return []

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)

        if norm > 1e-8:
            vec = vec / norm

        return vec.tolist()

    def _combine_vectors(
        self,
        a: List[float],
        b: List[float],
        wa: float,
        wb: float
    ) -> List[float]:
        """Combine two vectors with weights using numpy.

        Args:
            a: First vector
            b: Second vector
            wa: Weight for first vector
            wb: Weight for second vector

        Returns:
            Combined weighted vector
        """
        if not a and not b:
            return []
        if not a:
            return b
        if not b:
            return a

        # Convert to numpy arrays
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)

        # Pad shorter vector with zeros if needed
        max_dim = max(len(vec_a), len(vec_b))
        if len(vec_a) < max_dim:
            vec_a = np.pad(vec_a, (0, max_dim - len(vec_a)), mode='constant')
        if len(vec_b) < max_dim:
            vec_b = np.pad(vec_b, (0, max_dim - len(vec_b)), mode='constant')

        # Weighted combination
        result = wa * vec_a + wb * vec_b
        return result.tolist()

    def _exp_time_decay(
        self,
        ts: Any,
        now_ts: float,
        half_life_days: float
    ) -> float:
        """Exponential time decay with half-life in days"""
        if ts is None:
            return 1.0

        try:
            ts_float = float(ts)
        except Exception:
            return 1.0

        try:
            delta_days = max(0.0, (now_ts - ts_float) / 86400.0)
            return math.exp(-math.log(2) * (delta_days / float(half_life_days)))
        except Exception:
            return 1.0

    # --------------------------------------------------------------------- #
    # Interaction Stream Consumer
    # --------------------------------------------------------------------- #

    def _start_interaction_consumer(self):
        """Khởi động consumer để xử lý interaction events từ Redis stream."""
        if not self.redis_client:
            print("Warning: Redis not available, skipping interaction stream consumer")
            return

        def consume_loop():
            while True:
                try:
                    self._process_interaction_stream()
                except Exception as e:
                    print(f"Error in interaction stream consumer: {e}")
                    time.sleep(5)  # Đợi lâu hơn khi có lỗi

        thread = threading.Thread(target=consume_loop, daemon=True)
        thread.start()
        print("Interaction stream consumer started")

    def _process_interaction_stream(self):
        """Xử lý messages từ outbox-events stream với consumer group."""
        if not self.redis_client:
            return

        try:
            stream_name = Config.INTERACTION_STREAM_NAME
            group_name = Config.INTERACTION_CONSUMER_GROUP
            consumer_name = "recommend-consumer-1"

            # Tạo consumer group nếu chưa tồn tại
            try:
                self.redis_client.xgroup_create(
                    name=stream_name,
                    groupname=group_name,
                    id="0",  # Đọc từ đầu stream
                    mkstream=True
                )
                print(
                    f"Created consumer group '{group_name}' on stream '{stream_name}'")
            except Exception as e:
                # Group đã tồn tại
                if "BUSYGROUP" not in str(e):
                    raise

            # Đọc messages sử dụng consumer group
            messages = self.redis_client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                # ">" = chỉ đọc messages chưa được deliver
                streams={stream_name: ">"},
                count=10,
                block=1000  # Block 1 second
            )

            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    try:
                        # Xử lý event
                        self._handle_interaction_event(fields)

                        # Acknowledge message sau khi xử lý thành công
                        self.redis_client.xack(stream_name, group_name, msg_id)

                    except Exception as e:
                        print(
                            f"Error processing interaction event {msg_id}: {e}")
                        # Message sẽ được retry sau (pending)

        except Exception as e:
            if "NOGROUP" not in str(e):
                print(f"Warning: Interaction stream read error: {e}")

    def _handle_interaction_event(self, fields: Dict[str, bytes]):
        """Xử lý một interaction event từ outbox stream.

        Format từ backend Spring Boot:
        - aggregateType: "USER_INTERACTION" hoặc "JOB"
        - aggregateId: ID của user hoặc job
        - eventType: "CREATED", "APPLY", "SAVE", "CLICK", etc.
        - payload: JSON string chứa chi tiết event
        - occurredAt: Timestamp
        """
        try:
            # Decode fields từ bytes
            aggregate_type = fields.get(b'aggregateType', b'').decode('utf-8')
            aggregate_id = fields.get(b'aggregateId', b'').decode('utf-8')
            event_type = fields.get(b'eventType', b'').decode('utf-8')
            payload_str = fields.get(b'payload', b'{}').decode('utf-8')
            occurred_at = fields.get(b'occurredAt', b'').decode('utf-8')

            # Chỉ xử lý USER_INTERACTION events
            if aggregate_type != "USER_INTERACTION":
                print(
                    f"Skipping non-interaction event: aggregateType={aggregate_type}")
                return

            # Parse payload JSON
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                print(f"Invalid JSON payload: {payload_str[:100]}")
                return

            # Extract interaction data từ payload
            user_id = payload.get('userId') or payload.get('accountId')
            job_id = payload.get('jobId')
            interaction_type = payload.get('interactionType') or event_type
            timestamp = payload.get('timestamp') or occurred_at

            if not user_id or not job_id:
                print(f"Missing user_id or job_id in payload: {payload}")
                return

            # Validate interaction type
            interaction_type_upper = str(interaction_type).upper()
            if interaction_type_upper not in INTERACTION_WEIGHTS:
                print(f"Unknown interaction type: {interaction_type}")
                return

            if self._is_duplicate_interaction(aggregate_id):
                print(f"Skipping duplicate interaction: id={aggregate_id}")
                return

            # Update interactions cache
            self._update_user_interactions_cache(
                user_id=int(user_id),
                job_id=int(job_id),
                event_type=interaction_type_upper,
                timestamp=timestamp
            )

            # Invalidate short-term vector cache
            self.invalidate_short_term_cache(int(user_id))

            # Đánh dấu đã xử lý
            self._mark_interaction_processed(aggregate_id)

            print(
                f"Processed interaction: user={user_id}, job={job_id}, type={interaction_type_upper}")

        except Exception as e:
            print(f"Error handling interaction event: {e}")
            import traceback
            traceback.print_exc()

    def _update_user_interactions_cache(
        self,
        user_id: int,
        job_id: int,
        event_type: str,
        timestamp: Any
    ):
        """Cập nhật interactions cache trong Redis."""
        if not self.redis_client:
            return

        try:
            cache_key = f"user_interactions:{user_id}"

            # Lấy interactions hiện tại
            interactions = defaultdict(dict)
            cached = self.redis_client.get(cache_key)
            if cached:
                if isinstance(cached, (bytes, bytearray)):
                    cached = cached.decode()
                existing = json.loads(cached)
                for key, value in existing.items():
                    if isinstance(value, dict):
                        interactions[key] = value
                    elif isinstance(value, list):
                        interactions[key] = {int(v): None for v in value}

            # Thêm interaction mới
            event_type_upper = str(event_type).upper()
            if event_type_upper in INTERACTION_WEIGHTS:
                # Convert timestamp to float nếu cần
                ts_value = None
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            # Parse ISO format
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00"))
                            ts_value = dt.timestamp()
                        else:
                            ts_value = float(timestamp)
                    except Exception:
                        ts_value = datetime.now(timezone.utc).timestamp()
                else:
                    ts_value = datetime.now(timezone.utc).timestamp()

                interactions[event_type_upper][job_id] = ts_value

            # Lưu lại cache (TTL 7 ngày)
            self.redis_client.setex(
                cache_key,
                7 * 24 * 3600,
                json.dumps({k: dict(v) if isinstance(v, dict)
                           else v for k, v in interactions.items()})
            )

        except Exception as e:
            print(f"Error updating user interactions cache: {e}")

    def _is_duplicate_interaction(self, interaction_id: str) -> bool:
        """Kiểm tra interaction đã được xử lý chưa (deduplication).

        Args:
            interaction_id: ID của interaction event (aggregateId từ outbox)

        Returns:
            True nếu interaction đã được xử lý, False nếu chưa
        """
        if not self.redis_client or not interaction_id:
            return False

        try:
            key = f"processed_interaction:{interaction_id}"
            exists = self.redis_client.exists(key)
            return bool(exists)
        except Exception as e:
            print(f"Warning: Failed to check duplicate interaction: {e}")
            return False

    def _mark_interaction_processed(self, interaction_id: str) -> None:
        """Đánh dấu interaction đã được xử lý (TTL 7 ngày).

        Args:
            interaction_id: ID của interaction event (aggregateId từ outbox)
        """
        if not self.redis_client or not interaction_id:
            return

        try:
            key = f"processed_interaction:{interaction_id}"
            self.redis_client.setex(key, 7 * 24 * 3600, "1")
        except Exception as e:
            print(f"Warning: Failed to mark interaction as processed: {e}")

    # --------------------------------------------------------------------- #
    # Các helper mặc định (cần override hoặc tích hợp thực tế)
    # --------------------------------------------------------------------- #

    def _get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Lấy user profile từ API hoặc Milvus.

        Nếu chưa có user vector trên Milvus, gọi API để lấy profile.
        """
        # Kiểm tra xem đã có user vector trên Milvus chưa
        try:
            user_vector = self.milvus_service.get_user_vector(user_id)
            if user_vector:
                # Đã có vector, không cần gọi API
                return None
        except Exception:
            pass

        # Chưa có vector, gọi API lấy profile
        try:
            api_url = f"{Config.CANDIDATE_API_BASE_URL}/api/candidates/profile/{user_id}"
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 1000 and data.get("data"):
                    profile_data = data["data"]
                    # Chuẩn hóa format: educations -> education
                    if "educations" in profile_data:
                        profile_data["education"] = profile_data.pop(
                            "educations")
                    return profile_data
        except Exception as e:
            print(f"Warning: Failed to fetch user profile from API: {e}")

        return None

    def _get_user_interactions(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Lấy user interactions từ Redis cache (được update từ stream)."""
        if not self.redis_client:
            return {}

        try:
            cache_key = f"user_interactions:{user_id}"
            cached = self.redis_client.get(cache_key)
            if cached:
                if isinstance(cached, (bytes, bytearray)):
                    cached = cached.decode()
                return json.loads(cached)
        except Exception as e:
            print(f"Warning: Failed to get user interactions from cache: {e}")

        return {}

    def _get_popular_jobs(
        self,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Placeholder: trả về danh sách job phổ biến."""
        return []

    def reload_model(self) -> bool:
        """Force reload CF model (called by hot reload endpoint)"""
        try:
            logger.info("Force reloading CF model...")
            success = self._load_cf_model(force_reload=True)
            if success:
                logger.info("✓ Model reloaded successfully")
            else:
                logger.warning("Model reload failed or model unchanged")
            return success
        except Exception as e:
            logger.error(f"Reload error: {e}")
            return False
