"""Motion-Aware Adaptive Pooling module."""

from typing import List
import numpy as np
import torch
import torch.nn as nn
from sharingan.utils.flow import OpticalFlow


class MotionAwareAdaptivePooling(nn.Module):
    """Prioritize dynamic frames using optical flow."""

    def __init__(self, motion_threshold: float = 0.1):
        """
        Initialize Motion-Aware Adaptive Pooling.

        Args:
            motion_threshold: Threshold for significant motion (0-1)
        """
        super().__init__()
        self.motion_threshold = motion_threshold
        self.optical_flow = OpticalFlow()

        # Learnable pooling weights
        self.pooling_network = nn.Sequential(
            nn.Linear(1, 16),  # Motion score input
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output weight between 0 and 1
        )

    def compute_motion_score(self, frame_t: np.ndarray, frame_prev: np.ndarray) -> float:
        """
        Compute motion score between consecutive frames.

        Args:
            frame_t: Current frame (H, W, C)
            frame_prev: Previous frame (H, W, C)

        Returns:
            Motion score (0-1)
        """
        # Compute optical flow
        flow = self.optical_flow.compute_flow(frame_prev, frame_t, method="farneback")

        # Compute average magnitude
        magnitude = self.optical_flow.flow_magnitude(flow)

        # Normalize to 0-1 range (assuming max reasonable flow is 10 pixels)
        normalized_score = min(magnitude / 10.0, 1.0)

        return float(normalized_score)

    def should_process(self, motion_score: float) -> bool:
        """
        Determine if frame should be processed in detail.

        Args:
            motion_score: Motion score (0-1)

        Returns:
            True if frame should be processed
        """
        return motion_score > self.motion_threshold

    def forward(
        self,
        frames: List[np.ndarray],
        embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Pool embeddings with motion-aware weighting.

        Args:
            frames: List of video frames
            embeddings: List of frame embeddings

        Returns:
            Pooled embedding tensor
        """
        if len(frames) == 0 or len(embeddings) == 0:
            raise ValueError("Empty frames or embeddings")

        if len(frames) != len(embeddings):
            raise ValueError("Number of frames and embeddings must match")

        # Compute motion scores
        motion_scores = []
        for i in range(len(frames)):
            if i == 0:
                # First frame has no previous frame
                motion_scores.append(1.0)  # Always include first frame
            else:
                score = self.compute_motion_score(frames[i], frames[i-1])
                motion_scores.append(score)

        # Convert motion scores to weights using learned network
        motion_tensor = torch.tensor(motion_scores, dtype=torch.float32).reshape(-1, 1)
        weights = self.pooling_network(motion_tensor).squeeze(-1)  # (T,)

        # Normalize weights
        weights = weights / (weights.sum() + 1e-8)

        # Stack embeddings
        if embeddings[0].dim() == 1:
            embedding_stack = torch.stack(embeddings)  # (T, D)
        else:
            embedding_stack = torch.cat(embeddings, dim=0)  # (T, D)

        # Weighted pooling
        pooled = torch.sum(embedding_stack * weights.unsqueeze(-1), dim=0)

        return pooled

    def forward_adaptive(
        self,
        frames: List[np.ndarray],
        embeddings: List[torch.Tensor],
        process_fn=None
    ) -> List[torch.Tensor]:
        """
        Adaptively process frames based on motion.

        Frames with high motion are processed in detail,
        frames with low motion are skipped or processed lightly.

        Args:
            frames: List of video frames
            embeddings: List of frame embeddings (can be None for unprocessed frames)
            process_fn: Optional function to process frames with high motion

        Returns:
            List of processed embeddings
        """
        processed_embeddings = []

        for i in range(len(frames)):
            if i == 0:
                # Always process first frame
                if embeddings[i] is not None:
                    processed_embeddings.append(embeddings[i])
                elif process_fn is not None:
                    processed_embeddings.append(process_fn(frames[i]))
            else:
                # Compute motion score
                motion_score = self.compute_motion_score(frames[i], frames[i-1])

                if self.should_process(motion_score):
                    # High motion: process in detail
                    if embeddings[i] is not None:
                        processed_embeddings.append(embeddings[i])
                    elif process_fn is not None:
                        processed_embeddings.append(process_fn(frames[i]))
                else:
                    # Low motion: reuse previous embedding
                    if len(processed_embeddings) > 0:
                        processed_embeddings.append(processed_embeddings[-1])

        return processed_embeddings

    def get_motion_scores(self, frames: List[np.ndarray]) -> List[float]:
        """
        Compute motion scores for all frames.

        Args:
            frames: List of video frames

        Returns:
            List of motion scores
        """
        motion_scores = []
        for i in range(len(frames)):
            if i == 0:
                motion_scores.append(1.0)
            else:
                score = self.compute_motion_score(frames[i], frames[i-1])
                motion_scores.append(score)
        return motion_scores

    def __repr__(self) -> str:
        """String representation."""
        return f"MotionAwareAdaptivePooling(motion_threshold={self.motion_threshold})"
