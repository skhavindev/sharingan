"""Video loading with multi-backend support."""

from typing import Union, Iterator, Optional
import numpy as np
import cv2
from sharingan.exceptions import VideoLoadError


class VideoLoader:
    """Handles video file and stream loading with unified interface."""

    def __init__(self, source: Union[str, int], backend: str = "opencv"):
        """
        Initialize video loader.

        Args:
            source: File path, URL, or camera index
            backend: Video backend ("opencv", "decord", "pyav")

        Raises:
            VideoLoadError: If video source cannot be loaded
        """
        self.source = source
        self.backend = backend
        self._cap = None
        self._fps = None
        self._total_frames = None
        self._current_frame_idx = 0

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the video backend."""
        if self.backend == "opencv":
            self._initialize_opencv()
        elif self.backend == "decord":
            self._initialize_decord()
        elif self.backend == "pyav":
            self._initialize_pyav()
        else:
            raise VideoLoadError(
                f"Unsupported backend: {self.backend}. "
                f"Supported backends: opencv, decord, pyav"
            )

    def _initialize_opencv(self) -> None:
        """Initialize OpenCV backend."""
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise VideoLoadError(
                    f"Failed to open video source: {self.source}. "
                    f"Check if the file exists or the stream is accessible."
                )

            # Get video properties
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # For streams, frame count might be 0 or -1
            self._total_frames = total if total > 0 else None

        except Exception as e:
            raise VideoLoadError(f"Error initializing OpenCV backend: {str(e)}")

    def _initialize_decord(self) -> None:
        """Initialize Decord backend."""
        try:
            import decord
            decord.bridge.set_bridge("numpy")
            self._cap = decord.VideoReader(str(self.source))
            self._fps = self._cap.get_avg_fps()
            self._total_frames = len(self._cap)
        except ImportError:
            raise VideoLoadError(
                "Decord backend requires 'decord' package. "
                "Install with: pip install decord"
            )
        except Exception as e:
            raise VideoLoadError(f"Error initializing Decord backend: {str(e)}")

    def _initialize_pyav(self) -> None:
        """Initialize PyAV backend."""
        try:
            import av
            self._cap = av.open(str(self.source))
            stream = self._cap.streams.video[0]
            self._fps = float(stream.average_rate)
            self._total_frames = stream.frames if stream.frames > 0 else None
        except ImportError:
            raise VideoLoadError(
                "PyAV backend requires 'av' package. "
                "Install with: pip install av"
            )
        except Exception as e:
            raise VideoLoadError(f"Error initializing PyAV backend: {str(e)}")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        if self.backend == "opencv":
            return self._iter_opencv()
        elif self.backend == "decord":
            return self._iter_decord()
        elif self.backend == "pyav":
            return self._iter_pyav()

    def _iter_opencv(self) -> Iterator[np.ndarray]:
        """Iterate frames using OpenCV."""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._current_frame_idx += 1
            yield frame

    def _iter_decord(self) -> Iterator[np.ndarray]:
        """Iterate frames using Decord."""
        for i in range(len(self._cap)):
            frame = self._cap[i].asnumpy()
            self._current_frame_idx += 1
            yield frame

    def _iter_pyav(self) -> Iterator[np.ndarray]:
        """Iterate frames using PyAV."""
        for frame in self._cap.decode(video=0):
            frame_array = frame.to_ndarray(format="rgb24")
            self._current_frame_idx += 1
            yield frame_array

    def get_frame(self, index: int) -> np.ndarray:
        """
        Random access to specific frame.

        Args:
            index: Frame index to retrieve

        Returns:
            Frame as numpy array (H, W, C) in RGB format

        Raises:
            VideoLoadError: If frame cannot be retrieved
        """
        if self.backend == "opencv":
            return self._get_frame_opencv(index)
        elif self.backend == "decord":
            return self._get_frame_decord(index)
        elif self.backend == "pyav":
            raise VideoLoadError("Random access not supported with PyAV backend")

    def _get_frame_opencv(self, index: int) -> np.ndarray:
        """Get frame using OpenCV."""
        if self._total_frames and index >= self._total_frames:
            raise VideoLoadError(f"Frame index {index} out of range")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            raise VideoLoadError(f"Failed to read frame at index {index}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _get_frame_decord(self, index: int) -> np.ndarray:
        """Get frame using Decord."""
        if index >= len(self._cap):
            raise VideoLoadError(f"Frame index {index} out of range")

        return self._cap[index].asnumpy()

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def total_frames(self) -> Optional[int]:
        """Total frame count (None for streams)."""
        return self._total_frames

    def __del__(self):
        """Cleanup resources."""
        if self._cap is not None:
            if self.backend == "opencv":
                self._cap.release()
            elif self.backend in ["decord", "pyav"]:
                # Decord and PyAV handle cleanup automatically
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.__del__()
