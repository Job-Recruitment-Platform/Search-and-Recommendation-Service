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
    def extract_skill_names(skills: Any) -> str:
        """
        Extract skill names from skills array and return as comma-separated string.
        Handles both formats:
        - Array of strings: ["Python", "Java"] -> "Python, Java"
        - Array of objects: [{"id": 1, "name": "Python"}, {"id": 2, "name": "Java"}] -> "Python, Java"
        Returns VARCHAR-compatible string for Milvus
        """
        if not skills or not isinstance(skills, list):
            return ""
        
        skill_names = []
        
        if len(skills) > 0 and isinstance(skills[0], dict):
            # Array of objects: extract "name" field
            for skill in skills:
                if isinstance(skill, dict):
                    name = skill.get("name")
                    if name:
                        skill_names.append(str(name))
        else:
            # Array of strings: convert to list
            skill_names = [str(skill) for skill in skills if skill]
        
        # Join with comma and space, return as VARCHAR string
        return ", ".join(skill_names)

    @staticmethod
    def build_entity(dense_vec: List[float], sparse_vec: dict, job: Dict) -> Dict:
        """Build a structured entity from raw job data"""
        try:
            job_id = job.get("id")
            
            title = DataProcessor.clean_text(job.get("title", ""))
            skills = DataProcessor.extract_skill_names(job.get("skills", []))
            location = DataProcessor.clean_text(job.get("location", ""))
            description = DataProcessor.clean_text(job.get("description", ""))
            
            company = DataProcessor.clean_text(job.get("company", ""))
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
                "location": location,
                "description": description,
                "company": company,
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
                "sparse_vector": sparse_vec,
            }
            return entity
        except Exception as e:
            logger.error(f"Error building entity: {e}")
            return {}

    @staticmethod
    def build_entities(
        dense_vecs: List[List[float]],
        sparse_vecs,
        jobs: List[Dict]
    ) -> List[Dict]:
        """Build structured entities from raw job data."""
        entities: List[Dict] = []

        # Normalize scipy sparse matrix to list of {index: value} using COO triplets
        if hasattr(sparse_vecs, "shape"):
            coo = sparse_vecs.tocoo()
            rows = int(coo.shape[0])
            row_dicts = [{} for _ in range(rows)]
            for r, c, v in zip(coo.row, coo.col, coo.data):
                row_dicts[int(r)][int(c)] = float(v)
            sparse_vecs = row_dicts

        count =  len(jobs)
        for idx in range(count):
            entity = DataProcessor.build_entity(
                dense_vec=dense_vecs[idx],
                sparse_vec=sparse_vecs[idx],
                job=jobs[idx],
            )
            if entity:
                entities.append(entity)

        return entities
    
    @staticmethod
    def combine_job_text(job: Dict) -> str:
        """Combine only title, skills, location for sparse text embedding"""
        title = DataProcessor.clean_text(job.get("title", ""))
        skills = DataProcessor.extract_skill_names(job.get("skills", []))
        location = DataProcessor.clean_text(job.get("location", ""))
        combined_text = f"{title} {skills} {location}"
        return combined_text