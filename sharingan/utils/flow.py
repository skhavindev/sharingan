"""Optical flow computation utilities."""

import numpy as np
import cv2


class OpticalFlow:
    """Optical flow computation utilities."""

    @staticmethod
    def compute_flow(
        frame1: np.ndarray,
        frame2: np.ndarray,
        method: str = "farneback"
    ) -> np.ndarray:
        """
        Compute optical flow between frames.

        Args:
            frame1: First frame (H, W, C) or (H, W) in RGB or grayscale
            frame2: Second frame (H, W, C) or (H, W) in RGB or grayscale
            method: Flow computation method ("farneback", "lucas_kanade")

        Returns:
            Flow field of shape (H, W, 2) with (dx, dy) at each pixel

        Raises:
            ValueError: If method is not supported
        """
        # Convert to grayscale if needed
        if frame1.ndim == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1

        if frame2.ndim == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2

        # Ensure uint8 type
        if gray1.dtype != np.uint8:
            gray1 = (gray1 * 255).astype(np.uint8)
        if gray2.dtype != np.uint8:
            gray2 = (gray2 * 255).astype(np.uint8)

        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif method == "lucas_kanade":
            # For Lucas-Kanade, we compute dense flow using a grid of points
            # This is a simplified version
            flow = OpticalFlow._compute_dense_lk(gray1, gray2)
        else:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: farneback, lucas_kanade"
            )

        return flow

    @staticmethod
    def _compute_dense_lk(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow using Lucas-Kanade.

        Args:
            gray1: First grayscale frame
            gray2: Second grayscale frame

        Returns:
            Dense flow field
        """
        # Create a grid of points
        h, w = gray1.shape
        step = 10
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(np.float32)
        points = np.vstack([x, y]).T.reshape(-1, 1, 2)

        # Compute sparse flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, points, None, **lk_params
        )

        # Create dense flow field by interpolation
        flow = np.zeros((h, w, 2), dtype=np.float32)

        if new_points is not None and status is not None:
            good_old = points[status == 1]
            good_new = new_points[status == 1]

            if len(good_old) > 0:
                # Compute flow vectors
                flow_vectors = good_new - good_old

                # Interpolate to dense grid
                from scipy.interpolate import griddata
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

                flow[:, :, 0] = griddata(
                    good_old.reshape(-1, 2),
                    flow_vectors[:, 0, 0],
                    (grid_x, grid_y),
                    method='linear',
                    fill_value=0
                )
                flow[:, :, 1] = griddata(
                    good_old.reshape(-1, 2),
                    flow_vectors[:, 0, 1],
                    (grid_x, grid_y),
                    method='linear',
                    fill_value=0
                )

        return flow

    @staticmethod
    def flow_magnitude(flow: np.ndarray) -> float:
        """
        Compute average flow magnitude.

        Args:
            flow: Flow field of shape (H, W, 2)

        Returns:
            Average magnitude across all pixels
        """
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        return float(np.mean(magnitude))

    @staticmethod
    def flow_magnitude_map(flow: np.ndarray) -> np.ndarray:
        """
        Compute flow magnitude at each pixel.

        Args:
            flow: Flow field of shape (H, W, 2)

        Returns:
            Magnitude map of shape (H, W)
        """
        return np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

    @staticmethod
    def visualize_flow(flow: np.ndarray) -> np.ndarray:
        """
        Visualize optical flow as RGB image.

        Args:
            flow: Flow field of shape (H, W, 2)

        Returns:
            RGB visualization of shape (H, W, 3)
        """
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        # Compute magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Map angle to hue, magnitude to value
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb
