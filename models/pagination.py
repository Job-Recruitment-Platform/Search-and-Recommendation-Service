"""Pagination models"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PaginationInfo:
    """Pagination information"""
    limit: int
    offset: int
    total: Optional[int] = None  # Total number of items (if available)
    count: int = 0  # Number of items in current page
    has_next: Optional[bool] = None  # Whether there are more items
    has_prev: bool = False  # Whether there are previous items

    def to_dict(self) -> dict:
        """Convert to dict"""
        result = {
            "limit": self.limit,
            "offset": self.offset,
            "count": self.count,
            "hasPrev": self.has_prev,
        }
        if self.total is not None:
            result["total"] = self.total
        if self.has_next is not None:
            result["hasNext"] = self.has_next
        return result

