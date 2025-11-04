"""Job data models"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class JobSkill:
    """Skill model"""
    id: Optional[int] = None
    name: str = ""
    aliases: Optional[str] = None
    date_created: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "JobSkill":
        """Create JobSkill from dict"""
        if isinstance(data, dict):
            return cls(
                id=data.get("id"),
                name=data.get("name", ""),
                aliases=data.get("aliases"),
                date_created=data.get("dateCreated") or data.get("date_created"),
            )
        elif isinstance(data, str):
            return cls(name=data)
        else:
            return cls(name=str(data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        result = {"name": self.name}
        if self.id is not None:
            result["id"] = self.id
        if self.aliases:
            result["aliases"] = self.aliases
        if self.date_created:
            result["dateCreated"] = self.date_created
        return result


@dataclass
class Job:
    """Job model with all fields"""
    id: int
    title: str = ""
    company: str = ""
    description: str = ""
    job_role: str = ""
    seniority: str = ""
    location: str = ""
    work_mode: str = ""
    currency: str = ""
    status: str = ""
    skills: List[JobSkill] = field(default_factory=list)
    min_experience_years: int = 0
    salary_min: int = 0
    salary_max: int = 0
    max_candidates: int = 0
    date_posted: int = 0  # timestamp in milliseconds
    date_expires: int = 0  # timestamp in milliseconds

    @property
    def jobRole(self) -> str:
        return self.job_role

    @property
    def workMode(self) -> str:
        return self.work_mode

    @property
    def datePosted(self) -> int:
        return self.date_posted

    @property
    def dateExpires(self) -> int:
        return self.date_expires

    @property
    def minExperienceYears(self) -> int:
        return self.min_experience_years

    @property
    def salaryMin(self) -> int:
        return self.salary_min

    @property
    def salaryMax(self) -> int:
        return self.salary_max

    @property
    def maxCandidates(self) -> int:
        return self.max_candidates

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create Job from dict (handles both camelCase and snake_case)"""
        # Normalize skills
        skills_data = data.get("skills", [])
        skills = [JobSkill.from_dict(s) for s in skills_data] if skills_data else []

        # Parse dates if strings
        date_posted = cls._parse_date(
            data.get("date_posted") or data.get("datePosted")
        )
        date_expires = cls._parse_date(
            data.get("date_expires") or data.get("dateExpires")
        )

        return cls(
            id=int(data.get("id", 0)),
            title=data.get("title", ""),
            company=data.get("company", ""),
            description=data.get("description", ""),
            job_role=data.get("job_role") or data.get("jobRole", ""),
            seniority=data.get("seniority", ""),
            location=data.get("location", ""),
            work_mode=data.get("work_mode") or data.get("workMode", ""),
            currency=data.get("currency", ""),
            status=data.get("status", ""),
            skills=skills,
            min_experience_years=int(
                data.get("min_experience_years") or data.get("minExperienceYears", 0)
            ),
            salary_min=int(data.get("salary_min") or data.get("salaryMin", 0)),
            salary_max=int(data.get("salary_max") or data.get("salaryMax", 0)),
            max_candidates=int(
                data.get("max_candidates") or data.get("maxCandidates", 0)
            ),
            date_posted=date_posted,
            date_expires=date_expires,
        )

    @staticmethod
    def _parse_date(date_value: Any) -> int:
        """Parse date string to timestamp (milliseconds)"""
        if not date_value:
            return 0

        if isinstance(date_value, int):
            return date_value

        if isinstance(date_value, str):
            try:
                date_str = date_value.strip()
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"

                # Fix microseconds if present (max 6 digits)
                if "." in date_str and ("+" in date_str or "-" in date_str[-6:]):
                    dot_idx = date_str.index(".")
                    tz_idx = len(date_str)
                    for i in range(len(date_str) - 1, dot_idx, -1):
                        if date_str[i] in "+-" and i > dot_idx + 1:
                            tz_idx = i
                            break

                    if tz_idx < len(date_str):
                        before_dot = date_str[:dot_idx]
                        after_dot = date_str[dot_idx + 1 : tz_idx]
                        timezone = date_str[tz_idx:]

                        if len(after_dot) > 6:
                            after_dot = after_dot[:6]

                        date_str = f"{before_dot}.{after_dot}{timezone}"

                dt = datetime.fromisoformat(date_str)
                return int(dt.timestamp() * 1000)
            except Exception:
                return 0

        return 0

    def to_dict(self, camel_case: bool = False) -> Dict[str, Any]:
        """Convert to dict"""
        if camel_case:
            return {
                "id": self.id,
                "title": self.title,
                "company": self.company,
                "jobRole": self.job_role,
                "seniority": self.seniority,
                "location": self.location,
                "workMode": self.work_mode,
                "currency": self.currency,
                "status": self.status,
                "skills": [s.to_dict() for s in self.skills],
                "minExperienceYears": self.min_experience_years,
                "salaryMin": self.salary_min,
                "salaryMax": self.salary_max,
                "maxCandidates": self.max_candidates,
                "datePosted": self.date_posted,
                "dateExpires": self.date_expires,
            }
        else:
            return {
                "id": self.id,
                "title": self.title,
                "company": self.company,
                "job_role": self.job_role,
                "seniority": self.seniority,
                "location": self.location,
                "work_mode": self.work_mode,
                "currency": self.currency,
                "status": self.status,
                "skills": [s.to_dict() for s in self.skills],
                "min_experience_years": self.min_experience_years,
                "salary_min": self.salary_min,
                "salary_max": self.salary_max,
                "max_candidates": self.max_candidates,
                "date_posted": self.date_posted,
                "date_expires": self.date_expires,
            }

    def get_skill_names(self) -> List[str]:
        """Extract skill names as list of strings"""
        return [skill.name for skill in self.skills if skill.name]

