"""
=============================================================================
Trajectory Tracking & Visualization
=============================================================================
Maintains a history of ball positions and draws a fading trajectory trail
on video frames.  Optionally applies simple smoothing (moving-average) to
reduce jitter from frame-to-frame detection noise.

Usage:
    from tools.trajectory import TrajectoryTracker

    tracker = TrajectoryTracker(config)
    tracker.update(frame_idx, ball_center, pos_3d)
    frame  = tracker.draw(frame, config)
=============================================================================
"""

import math
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np


class TrajectoryTracker:
    """
    Keeps a rolling window of ball positions and draws a fading polyline
    trail on each video frame.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise the tracker from the project config.

        Config keys used:
            trajectory.trail_length          - how many past points to show
            trajectory.smoothing_window      - moving-average window (0 = off)
            trajectory.line_thickness        - polyline thickness
            trajectory.color                 - BGR trail colour list
            trajectory.fade                  - whether to fade older points
        """
        traj_cfg = config.get("trajectory", {})

        self.trail_length: int = int(traj_cfg.get("trail_length", 30))
        self.smoothing_window: int = int(traj_cfg.get("smoothing_window", 0))
        self.line_thickness: int = int(traj_cfg.get("line_thickness", 2))
        self.color: Tuple[int, int, int] = tuple(
            traj_cfg.get("color", [0, 255, 255])
        )
        self.fade: bool = bool(traj_cfg.get("fade", True))

        # Internal state
        self._history: deque = deque(maxlen=self.trail_length)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        ball_center: Optional[List[float]],
        pos_3d: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a new observation.

        Args:
            frame_idx:   Current frame index.
            ball_center: [cx, cy] pixel coordinates, or ``None`` if not
                         detected this frame.
            pos_3d:      3D position dict from ``DepthEstimator.estimate()``,
                         or ``None``.
        """
        if ball_center is not None:
            self._history.append(
                {
                    "frame_idx": frame_idx,
                    "center": (float(ball_center[0]), float(ball_center[1])),
                    "pos_3d": pos_3d,
                }
            )

    def draw(
        self,
        frame: np.ndarray,
        config: Dict[str, Any],
    ) -> np.ndarray:
        """
        Draw the trajectory trail on *frame* (modified in-place).

        The trail is drawn as a polyline connecting the last N detected
        ball centres.  When *fade* is enabled the colour intensity
        decreases for older points.

        Args:
            frame:  BGR image array.
            config: Project config (used only for potential overrides).

        Returns:
            Frame with trajectory overlay.
        """
        points = self.get_trail_points()

        if len(points) < 2:
            return frame

        if not self.fade:
            # Simple solid polyline
            pts_array = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                frame,
                [pts_array],
                isClosed=False,
                color=self.color,
                thickness=self.line_thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            # Draw segment-by-segment with fading alpha
            n = len(points)
            for i in range(n - 1):
                # alpha goes from 0.2 (oldest) to 1.0 (newest)
                alpha = 0.2 + 0.8 * (i / max(n - 2, 1))
                seg_color = tuple(int(c * alpha) for c in self.color)
                thickness = max(
                    1, int(self.line_thickness * alpha)
                )
                cv2.line(
                    frame,
                    (int(points[i][0]), int(points[i][1])),
                    (int(points[i + 1][0]), int(points[i + 1][1])),
                    seg_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # Draw a small filled circle at the latest position
        latest = points[-1]
        cv2.circle(
            frame,
            (int(latest[0]), int(latest[1])),
            radius=5,
            color=self.color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        return frame

    def get_trail_points(self) -> List[Tuple[float, float]]:
        """
        Return the list of (cx, cy) pixel coordinates currently in the
        trail history (oldest first).

        If smoothing_window > 0, returns smoothed coordinates.
        """
        raw = [entry["center"] for entry in self._history]

        if self.smoothing_window <= 1 or len(raw) < 2:
            return raw

        return self._smooth(raw, self.smoothing_window)

    def get_full_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the full internal history buffer."""
        return list(self._history)

    def reset(self) -> None:
        """Clear the trajectory history."""
        self._history.clear()

    @property
    def length(self) -> int:
        """Number of points currently stored."""
        return len(self._history)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _smooth(
        points: List[Tuple[float, float]], window: int
    ) -> List[Tuple[float, float]]:
        """Apply a centred moving-average to the point list."""
        n = len(points)
        half = window // 2
        smoothed: List[Tuple[float, float]] = []

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            xs = [p[0] for p in points[lo:hi]]
            ys = [p[1] for p in points[lo:hi]]
            smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        return smoothed
