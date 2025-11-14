"""Event detection with semantic understanding."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.signal import find_peaks


@dataclass
class Event:
    """Represents a detected event."""
    event_id: str
    event_type: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    description: str
    confidence: float
    involved_entities: List[str]
    embedding: np.ndarray


class EventDetector:
    """Detect significant events in video sequences."""

    def __init__(self, sensitivity: float = 0.5):
        """
        Initialize event detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1). Lower = more events detected
        """
        self.sensitivity = sensitivity

    def detect_events(
        self, 
        embeddings: np.ndarray, 
        timestamps: List[float],
        frame_indices: Optional[List[int]] = None,
        descriptions: Optional[List[str]] = None
    ) -> List[Event]:
        """
        Detect events from temporal embeddings.
        
        Args:
            embeddings: Frame embeddings (N, D)
            timestamps: Frame timestamps
            frame_indices: Optional frame indices
            descriptions: Optional frame descriptions
            
        Returns:
            List of detected events
        """
        if len(embeddings) < 2:
            return []
        
        if frame_indices is None:
            frame_indices = list(range(len(embeddings)))
        
        # Detect scene changes
        scene_changes = self.detect_scene_changes(embeddings)
        
        # Create events from scene changes
        events = []
        for i, change_idx in enumerate(scene_changes):
            # Determine event boundaries
            start_idx = change_idx
            end_idx = min(change_idx + 5, len(embeddings) - 1)
            
            # Get embedding difference magnitude as confidence
            if change_idx > 0:
                diff = np.linalg.norm(embeddings[change_idx] - embeddings[change_idx - 1])
                mean_diff = np.mean([np.linalg.norm(embeddings[j] - embeddings[j-1]) 
                                    for j in range(1, len(embeddings))])
                confidence = min(diff / (mean_diff + 1e-8), 1.0)
            else:
                confidence = 0.8
            
            # Create event
            event = Event(
                event_id=f"event_{i:03d}",
                event_type="scene_change",
                start_frame=frame_indices[start_idx],
                end_frame=frame_indices[end_idx],
                start_time=timestamps[start_idx],
                end_time=timestamps[end_idx],
                description=descriptions[change_idx] if descriptions else f"Scene transition detected",
                confidence=float(confidence),
                involved_entities=[],
                embedding=embeddings[change_idx]
            )
            events.append(event)
        
        return events

    def detect_scene_changes(self, embeddings: np.ndarray) -> List[int]:
        """
        Detect scene transition frames using embedding similarity.
        
        Args:
            embeddings: Frame embeddings (N, D)
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(embeddings) < 2:
            return []
        
        # Compute frame-to-frame cosine distances
        similarities = []
        for i in range(1, len(embeddings)):
            # Cosine similarity
            sim = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Convert similarity to distance
        distances = 1 - similarities
        
        # Adaptive threshold based on sensitivity
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (1 - self.sensitivity) * std_dist
        
        # Find peaks in distance (scene changes)
        peaks, properties = find_peaks(
            distances, 
            height=threshold,
            distance=10  # Minimum 10 frames between scene changes
        )
        
        # Convert to frame indices (add 1 because we computed differences)
        scene_changes = (peaks + 1).tolist()
        
        return scene_changes
    
    def detect_motion_events(self, embeddings: np.ndarray, motion_scores: List[float]) -> List[int]:
        """
        Detect motion-based events.
        
        Args:
            embeddings: Frame embeddings
            motion_scores: Motion scores for each frame
            
        Returns:
            List of frame indices with significant motion
        """
        if len(motion_scores) < 2:
            return []
        
        motion_array = np.array(motion_scores)
        threshold = np.mean(motion_array) + np.std(motion_array)
        
        # Find peaks in motion
        peaks, _ = find_peaks(motion_array, height=threshold, distance=5)
        
        return peaks.tolist()
