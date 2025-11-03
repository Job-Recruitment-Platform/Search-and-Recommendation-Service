"""Data processing utilities"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing utilities"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def extract_skill_names(skills: Any) -> List[str]:
        """
        Extract skill names from skills array.
        Handles both formats:
        - Array of strings: ["Python", "Java"]
        - Array of objects: [{"id": 1, "name": "Python"}, {"id": 2, "name": "Java"}]
        """
        if not skills or not isinstance(skills, list):
            return []
        
        # Check if first item is a dict (object) or string
        if isinstance(skills[0], dict):
            # Array of objects: extract "name" field
            skill_names = []
            for skill in skills:
                if isinstance(skill, dict):
                    name = skill.get("name")  # Get "name" field from skill object
                    if name:
                        skill_names.append(str(name))
            return skill_names
        else:
            # Array of strings: return as is
            return [str(skill) for skill in skills if skill]

    @staticmethod
    def _sparse_to_dict(sparse_vec: Any) -> Dict[int, float]:
        """Convert a sparse vector to {index: value} dict acceptable by Milvus."""
        try:
            if isinstance(sparse_vec, dict):
                return {int(k): float(v) for k, v in sparse_vec.items()}
            if hasattr(sparse_vec, 'tocoo'):
                coo = sparse_vec.tocoo()
                return {int(i): float(v) for i, v in zip(coo.col, coo.data)}
            if hasattr(sparse_vec, '__iter__') and not isinstance(sparse_vec, str):
                out: Dict[int, float] = {}
                for i, v in enumerate(sparse_vec):
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv != 0.0:
                        out[i] = fv
                return out
        except Exception:
            return {}
        return {}

    @staticmethod
    def build_entity(dense_vec: List[float], sparse_vec: Any, job: Dict) -> Dict:
        """Build a structured entity from raw job data"""
        try:
            job_id = job.get("id")
            title = DataProcessor.clean_text(job.get("title", ""))
            skills = DataProcessor.extract_skill_names(job.get("skills", []))
            company = DataProcessor.clean_text(job.get("company", ""))
            location = DataProcessor.clean_text(job.get("location", ""))
            
            job_role = DataProcessor.clean_text(job.get("job_role", ""))
            seniority = DataProcessor.clean_text(job.get("seniority", ""))
            min_experience_years = job.get("min_experience_years", 0)
            work_mode = DataProcessor.clean_text(job.get("work_mode", ""))
            salary_min = job.get("salary_min", 0)
            salary_max = job.get("salary_max", 0)
            currency = DataProcessor.clean_text(job.get("currency", ""))
            status = DataProcessor.clean_text(job.get("status", ""))
            
            max_candidates = job.get("max_candidates", 0)
            date_posted = job.get("date_posted", 0)
            date_expires = job.get("date_expires", 0)
            
            
            
            
            entity = {
                "id": job_id,
                "title": title,
                "skills": skills,
                "company": company,
                "location": location,
                "job_role": job_role,
                "seniority": seniority,
                "min_experience_years": min_experience_years,
                "work_mode": work_mode,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "currency": currency,
                "status": status,
                "max_candidates": max_candidates,
                "date_posted": date_posted,
                "date_expires": date_expires,
                "dense_vector": dense_vec,
                "sparse_vector": DataProcessor._sparse_to_dict(sparse_vec) if sparse_vec is not None else {},
            }
            return entity
        except Exception as e:
            logger.error(f"Error building entity: {e}")
            return {}

    @staticmethod
    def build_entities(dense_vecs: List[List[float]], sparse_vecs: Any, jobs: List[Dict]) -> List[Dict]:
        """Build structured entities from raw job data.
        sparse_vecs can be a list, a single sparse object, or None.
        """
        entities: List[Dict] = []
        count = min(len(dense_vecs), len(jobs))
        is_list = isinstance(sparse_vecs, list)
        for idx in range(count):
            if is_list:
                sv = sparse_vecs[idx] if idx < len(sparse_vecs) else None
            else:
                sv = sparse_vecs if sparse_vecs is not None and idx == 0 else None
            entity = DataProcessor.build_entity(
                dense_vec=dense_vecs[idx],
                sparse_vec=sv,
                job=jobs[idx],
            )
            if entity:
                entities.append(entity)
        return entities
    
    @staticmethod
    def combine_job_text(job: Dict) -> str:
        """Combine relevant job fields into a single text string for embedding generation"""
        title = DataProcessor.clean_text(job.get("title", ""))
        skills = " ".join(DataProcessor.extract_skill_names(job.get("skills", [])))
        company = DataProcessor.clean_text(job.get("company", ""))
        location = DataProcessor.clean_text(job.get("location", ""))
        
        combined_text = f"{title} {skills} {company} {location}"
        return combined_text