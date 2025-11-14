"""Natural language query (stub - to be implemented)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class QueryPlan:
    """Parsed query execution plan."""
    query_type: str
    query_embedding: np.ndarray
    temporal_constraints: Optional[Tuple[float, float]]
    spatial_constraints: Optional[Dict]
    entity_filters: List[str]


class NaturalLanguageQuery:
    """Natural language query interface."""

    def __init__(self, text_encoder: any):
        """Initialize with text encoder."""
        self.text_encoder = text_encoder

    def parse_query(self, query: str) -> QueryPlan:
        """Parse natural language query."""
        raise NotImplementedError("Query parsing to be implemented")

    def execute(self, query: str, video_context: any) -> List[any]:
        """Execute query against video context."""
        raise NotImplementedError("Query execution to be implemented")
