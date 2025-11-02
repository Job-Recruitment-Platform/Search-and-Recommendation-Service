"""Search weights model for hybrid search"""
from dataclasses import dataclass


@dataclass
class SearchWeights:
    """Search weights for hybrid search"""
    dense: float = 1.0
    sparse: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "SearchWeights":
        """Create from dict"""
        return cls(
            dense=float(data.get("dense", 1.0)),
            sparse=float(data.get("sparse", 1.0)),
        )

    def to_dict(self) -> dict:
        """Convert to dict"""
        return {"dense": self.dense, "sparse": self.sparse}

