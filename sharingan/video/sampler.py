"""Frame sampling strategies for video processing."""

from typing import Iterator, Tuple
import numpy as np


class FrameSampler:
    """Adaptive frame sampling strategies."""

    def __init__(self, strategy: str = "uniform", target_fps: float = 5.0):
        """
        Initialize frame sampler.

        Args:
            strategy: Sampling strategy ("uniform", "adaptive", "motion_based")
            target_fps: Target sampling rate in frames per second

        Raises:
            ValueError: If strategy is not supported
        """
        self.strategy = strategy
        self.target_fps = target_fps
        self._prev_frame = None
        self._frame_count = 0

        if strategy not in ["uniform", "adaptive", "motion_based"]:
            raise ValueError(
                f"Unsupported strategy: {strategy}. "
                f"Supported strategies: uniform, adaptive, motion_based"
            )

    def sample(
        self, frames: Iterator[np.ndarray], source_fps: float = 30.0
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Sample frames from iterator.

        Args:
            frames: Iterator of video frames
            source_fps: Source video FPS

        Yields:
            Tuple of (frame_index, frame)
        """
        if self.strategy == "uniform":
            yield from self._sample_uniform(frames, source_fps)
        elif self.strategy == "adaptive":
            yield from self._sample_adaptive(frames, source_fps)
        elif self.strategy == "motion_based":
            yield from self._sample_motion_based(frames, source_fps)

    def _sample_uniform(
        self, frames: Iterator[np.ndarray], source_fps: float
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Uniform sampling at target FPS.

        Args:
            frames: Iterator of video frames
            source_fps: Source video FPS

        Yields:
            Tuple of (frame_index, frame)
        """
        # Calculate frame skip interval
        skip_interval = max(1, int(source_fps / self.target_fps))

        frame_idx = 0
        for frame in frames:
            if frame_idx % skip_interval == 0:
                yield (frame_idx, frame)
            frame_idx += 1

    def _sample_adaptive(
        self, frames: Iterator[np.ndarray], source_fps: float
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Adaptive sampling based on frame differences.

        Samples more frequently when frames change significantly.

        Args:
            frames: Iterator of video frames
            source_fps: Source video FPS

        Yields:
            Tuple of (frame_index, frame)
        """
        skip_interval = max(1, int(source_fps / self.target_fps))
        diff_threshold = 0.1  # Threshold for significant change

        frame_idx = 0
        frames_since_last_sample = 0

        for frame in frames:
            should_sample = False

            # Always sample first frame
            if self._prev_frame is None:
                should_sample = True
            else:
                # Check if enough frames have passed
                if frames_since_last_sample >= skip_interval:
                    should_sample = True
                else:
                    # Check for significant change
                    diff = self._compute_frame_difference(self._prev_frame, frame)
                    if diff > diff_threshold:
                        should_sample = True

            if should_sample:
                yield (frame_idx, frame)
                self._prev_frame = frame.copy()
                frames_since_last_sample = 0
            else:
                frames_since_last_sample += 1

            frame_idx += 1

    def _sample_motion_based(
        self, frames: Iterator[np.ndarray], source_fps: float
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Motion-based sampling using optical flow.

        Samples frames with significant motion more frequently.

        Args:
            frames: Iterator of video frames
            source_fps: Source video FPS

        Yields:
            Tuple of (frame_index, frame)
        """
        skip_interval = max(1, int(source_fps / self.target_fps))
        motion_threshold = 0.05  # Threshold for significant motion

        frame_idx = 0
        frames_since_last_sample = 0

        for frame in frames:
            should_sample = False

            # Always sample first frame
            if self._prev_frame is None:
                should_sample = True
            else:
                # Compute motion score
                motion_score = self._compute_motion_score(self._prev_frame, frame)

                # Sample if motion is significant or interval reached
                if motion_score > motion_threshold or frames_since_last_sample >= skip_interval:
                    should_sample = True

            if should_sample:
                yield (frame_idx, frame)
                self._prev_frame = frame.copy()
                frames_since_last_sample = 0
            else:
                frames_since_last_sample += 1

            frame_idx += 1

    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute normalized difference between frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Normalized difference score (0-1)
        """
        # Resize for faster computation if needed
        if frame1.shape[0] > 480:
            import cv2
            h, w = 480, int(480 * frame1.shape[1] / frame1.shape[0])
            frame1 = cv2.resize(frame1, (w, h))
            frame2 = cv2.resize(frame2, (w, h))

        # Compute mean absolute difference
        diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
        return np.mean(diff) / 255.0

    def _compute_motion_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute motion score between frames using simple difference.

        For full optical flow, use utils.OpticalFlow.

        Args:
            frame1: Previous frame
            frame2: Current frame

        Returns:
            Motion score (0-1)
        """
        # Simple motion estimation using frame difference
        # For production, this would use optical flow from utils
        return self._compute_frame_difference(frame1, frame2)

    def reset(self) -> None:
        """Reset sampler state."""
        self._prev_frame = None
        self._frame_count = 0
