"""Entity tracking (stub - to be implemented)."""

from typing import List


class EntityTracker:
    """Track entities across video frames."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """Initialize entity tracker."""
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections: List[any], frame_idx: int) -> List[any]:
        """Update tracks with new detections."""
        raise NotImplementedError("Entity tracking to be implemented")

    def get_active_tracks(self) -> List[any]:
        """Get currently active tracks."""
        raise NotImplementedError("Entity tracking to be implemented")
