"""Entity data structures (stub - to be implemented)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """Detection representation."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    embedding: np.ndarray


@dataclass
class Track:
    """Represents an entity track across frames."""
    track_id: str
    entity_type: str
    bounding_boxes: List[Tuple[int, int, int, int]]
    frame_indices: List[int]
    embeddings: List[np.ndarray]
    confidence_scores: List[float]

    def get_trajectory(self) -> np.ndarray:
        """Get center point trajectory."""
        raise NotImplementedError("Trajectory computation to be implemented")

    def get_at_frame(self, frame_idx: int) -> Optional[Detection]:
        """Get detection at specific frame."""
        raise NotImplementedError("Frame retrieval to be implemented")
