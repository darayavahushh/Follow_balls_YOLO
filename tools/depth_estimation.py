"""
=============================================================================
Depth / 3D Position Estimation
=============================================================================
Estimates the 3D position (X, Y, Z) and distance of the football relative
to the camera using the pinhole camera model and a known reference object
size (standard FIFA size-5 football, diameter 22 cm).

Camera: Intel RealSense D435i RGB sensor.

Method ("size_reference"):
    Z  = (f * D_real) / d_px          depth along optical axis
    X  = (u - cx) * Z / fx            horizontal offset
    Y  = (v - cy) * Z / fy            vertical offset
    dist = sqrt(X² + Y² + Z²)         Euclidean distance to camera

Usage:
    from tools.depth_estimation import DepthEstimator

    estimator = DepthEstimator(config)
    result = estimator.estimate(ball_bbox, frame_shape)
=============================================================================
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class DepthEstimator:
    """
    Estimates 3D ball position from a single RGB frame using the
    pinhole projection model and known ball diameter.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise from the project configuration dictionary.

        Expected config keys used:
            camera.intrinsics.{fx, fy, cx, cy, width, height}
            camera.reference_objects.football_diameter_cm
        """
        cam = config["camera"]["intrinsics"]

        self.fx: float = float(cam["fx"])
        self.fy: float = float(cam["fy"])
        self.cx: float = float(cam["cx"])
        self.cy: float = float(cam["cy"])
        self.cal_width: int = int(cam["width"])
        self.cal_height: int = int(cam["height"])

        # Real-world ball diameter in **metres**
        diameter_cm = float(
            config["camera"]["reference_objects"]["football_diameter_cm"]
        )
        self.ball_diameter_m: float = diameter_cm / 100.0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def estimate(
        self,
        ball_bbox: List[float],
        frame_shape: Tuple[int, int],
    ) -> Dict[str, float]:
        """
        Estimate the 3D position of the ball.

        Args:
            ball_bbox: [x1, y1, x2, y2] bounding-box in pixel coords
                       of the **current frame resolution**.
            frame_shape: (height, width) of the current frame (used to
                         scale intrinsics when the video resolution
                         differs from the calibration resolution).

        Returns:
            Dictionary with keys:
                x_m, y_m, z_m    - 3D position in metres (camera frame)
                distance_m       - Euclidean distance in metres
                bbox_diameter_px - apparent diameter used (pixels)
        """
        x1, y1, x2, y2 = ball_bbox
        frame_h, frame_w = frame_shape

        # Scale intrinsics to actual frame resolution
        sx = frame_w / self.cal_width
        sy = frame_h / self.cal_height
        fx = self.fx * sx
        fy = self.fy * sy
        cx = self.cx * sx
        cy = self.cy * sy

        # Use the average of bbox width & height as apparent diameter
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        d_px = (bbox_w + bbox_h) / 2.0

        if d_px <= 0:
            return self._empty_result()

        # Focal length: average of fx, fy
        f = (fx + fy) / 2.0

        # Depth along optical axis
        z = (f * self.ball_diameter_m) / d_px

        # Centre of the bounding box in pixels
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        # 3D position
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        distance = math.sqrt(x * x + y * y + z * z)

        return {
            "x_m": round(x, 4),
            "y_m": round(y, 4),
            "z_m": round(z, 4),
            "distance_m": round(distance, 4),
            "bbox_diameter_px": round(d_px, 2),
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _empty_result() -> Dict[str, float]:
        """Return an empty / NaN result when estimation is not possible."""
        return {
            "x_m": float("nan"),
            "y_m": float("nan"),
            "z_m": float("nan"),
            "distance_m": float("nan"),
            "bbox_diameter_px": 0.0,
        }

    @staticmethod
    def format_position_str(result: Dict[str, float]) -> str:
        """
        Pretty-print a position result for overlay / logging.

        Args:
            result: Dictionary returned by ``estimate()``.

        Returns:
            Formatted string, e.g. "X=0.12 Y=-0.34 Z=5.60  D=5.62 m"
        """
        if math.isnan(result.get("distance_m", float("nan"))):
            return "3D: N/A"

        return (
            f"X={result['x_m']:+.2f} "
            f"Y={result['y_m']:+.2f} "
            f"Z={result['z_m']:.2f}  "
            f"D={result['distance_m']:.2f} m"
        )
