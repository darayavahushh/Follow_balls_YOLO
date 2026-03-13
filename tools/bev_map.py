"""
=============================================================================
Bird's Eye View (BEV) / 2D Top-View Map
=============================================================================
Provides two components for generating a camera-independent top-view map:

1.  **CameraMotionEstimator** - estimates frame-to-frame camera ego-motion
    using sparse optical-flow + Essential matrix decomposition (RANSAC).
    Accumulates a world ↔ camera pose so that 3D ball positions can be
    transformed into a fixed *world* coordinate frame that is **not
    affected by camera movement or shaking**.

2.  **BEVMapRenderer** - renders a clean top-down 2D map showing the ball
    position, ball trajectory, camera position, and camera trail with
    auto-scaling, grid lines, a scale bar, and a legend.

Camera model:   Intel RealSense D435i RGB (pinhole, known intrinsics).
Scale source:   Ball depth estimated via the known-diameter method.

Usage:
    from tools.bev_map import CameraMotionEstimator, BEVMapRenderer

    motion = CameraMotionEstimator(config)
    bev    = BEVMapRenderer(config)

    motion.update(frame, ball_depth=z)
    ball_world = motion.transform_to_world(pos_3d)
    bev.update(motion.get_camera_world_pos(), ball_world)
    map_img = bev.render()
=============================================================================
"""

import math
from typing import Dict, Any, Optional, List, Tuple, Callable

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# CAMERA MOTION ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class CameraMotionEstimator:
    """
    Estimates cumulative camera pose from monocular video using sparse
    optical flow and Essential-matrix decomposition.

    The accumulated world-to-camera transform ``(R_wc, t_wc)`` satisfies
    ``P_cam = R_wc @ P_world + t_wc``, so the inverse gives:

        P_world = R_wc^T @ (P_cam - t_wc)

    Scale is resolved using the ball's estimated depth (from the
    known-diameter method).  When the ball is not visible the last
    known depth is kept.
    """

    def __init__(self, config: Dict[str, Any]):
        # Camera intrinsics (calibration resolution)
        cam = config["camera"]["intrinsics"]
        self.fx = float(cam["fx"])
        self.fy = float(cam["fy"])
        self.cx = float(cam["cx"])
        self.cy = float(cam["cy"])
        self.cal_w = int(cam["width"])
        self.cal_h = int(cam["height"])

        # BEV motion config
        bev_motion = config.get("bev", {}).get("motion", {})
        self.min_features: int = int(bev_motion.get("min_features", 30))
        self.translation_threshold: float = float(
            bev_motion.get("translation_threshold", 0.005)
        )

        # Feature detection parameters (Shi-Tomasi corners)
        self._feature_params = dict(
            maxCorners=300,
            qualityLevel=float(bev_motion.get("feature_quality", 0.01)),
            minDistance=int(bev_motion.get("feature_min_distance", 20)),
            blockSize=7,
        )

        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        # ── cumulative world-to-camera transform ──
        self.R_wc: np.ndarray = np.eye(3, dtype=np.float64)
        self.t_wc: np.ndarray = np.zeros(3, dtype=np.float64)

        # Previous-frame state
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

        # Depth scale tracking
        self._last_depth: float = 5.0  # reasonable default (metres)

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _scaled_K(self, h: int, w: int) -> np.ndarray:
        """Camera matrix scaled to the actual frame resolution."""
        sx = w / self.cal_w
        sy = h / self.cal_h
        return np.array(
            [
                [self.fx * sx, 0.0, self.cx * sx],
                [0.0, self.fy * sy, self.cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners."""
        kps = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        return kps

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, ball_depth: Optional[float] = None) -> None:
        """
        Process a new frame and update the cumulative camera pose.

        Args:
            frame:      BGR image (current frame).
            ball_depth: Ball depth (Z) in metres from the depth estimator,
                        or ``None`` if the ball was not detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        K = self._scaled_K(h, w)

        # Update depth scale when available
        if (
            ball_depth is not None
            and ball_depth > 0
            and not math.isnan(ball_depth)
        ):
            # Smooth toward the new observation
            self._last_depth = 0.7 * self._last_depth + 0.3 * ball_depth

        # First frame – just initialise
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        # Not enough features in previous frame
        if self._prev_kps is None or len(self._prev_kps) < self.min_features:
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        # ── optical flow tracking ──
        new_kps, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_kps, None, **self._lk_params
        )

        if new_kps is None:
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        mask = status.flatten() == 1
        pts_old = self._prev_kps[mask].reshape(-1, 2)
        pts_new = new_kps[mask].reshape(-1, 2)

        if len(pts_old) < 8:
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        # ── Essential matrix ──
        E, inlier_mask = cv2.findEssentialMat(
            pts_old, pts_new, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or E.shape != (3, 3):
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        n_inliers, R, t, _ = cv2.recoverPose(
            E, pts_old, pts_new, K, mask=inlier_mask
        )

        if n_inliers < 5:
            self._prev_gray = gray
            self._prev_kps = self._detect_features(gray)
            return

        # ── compute metric scale ──
        # Median pixel displacement × depth / focal-length ≈ real translation
        flow = pts_new - pts_old
        median_flow = float(np.median(np.linalg.norm(flow, axis=1)))
        f_avg = (K[0, 0] + K[1, 1]) / 2.0
        scale = median_flow * self._last_depth / f_avg

        # ── accumulate pose ──
        # P_cam_new = R @ P_cam_old + t  ⇒  world-to-camera compounds as:
        #   R_wc(new) = R @ R_wc(old)
        #   t_wc(new) = R @ t_wc(old) + t * scale
        self.R_wc = R @ self.R_wc

        if scale >= self.translation_threshold:
            self.t_wc = R @ self.t_wc + t.flatten() * scale
        else:
            self.t_wc = R @ self.t_wc  # pure rotation – no translation

        # ── prepare for next frame ──
        self._prev_gray = gray
        self._prev_kps = self._detect_features(gray)

    def get_camera_world_pos(self) -> np.ndarray:
        """
        Camera position in the world frame (3-element array).

        Derived from  P_cam = R_wc · P_world + t_wc
        ⇒  camera origin (P_cam=0):  P_world = -R_wc^T · t_wc
        """
        return -self.R_wc.T @ self.t_wc

    def transform_to_world(
        self, pos_3d: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """
        Transform a point from camera frame to world frame.

        Args:
            pos_3d: Dict with keys ``x_m, y_m, z_m`` (camera frame),
                    or ``None``.

        Returns:
            Dict with world-frame ``x_m, y_m, z_m``, or ``None``.
        """
        if pos_3d is None:
            return None

        x = pos_3d.get("x_m", 0.0)
        y = pos_3d.get("y_m", 0.0)
        z = pos_3d.get("z_m", 0.0)

        if any(math.isnan(v) for v in (x, y, z)):
            return None

        p_cam = np.array([x, y, z], dtype=np.float64)
        p_world = self.R_wc.T @ (p_cam - self.t_wc)

        return {
            "x_m": float(p_world[0]),
            "y_m": float(p_world[1]),
            "z_m": float(p_world[2]),
        }


# ═══════════════════════════════════════════════════════════════════════
# BEV MAP RENDERER
# ═══════════════════════════════════════════════════════════════════════

class BEVMapRenderer:
    """
    Renders a clean top-down 2D map with:
      • Ball position & fading trajectory trail   (green)
      • Camera position & fading trail             (yellow-orange)
      • Metric grid lines with auto-spacing
      • Scale bar & legend

    World **X → map horizontal** (left/right).
    World **Z → map vertical**   (forward = up on map).
    """

    def __init__(self, config: Dict[str, Any]):
        bev = config.get("bev", {})
        self.map_size: int = int(bev.get("map_size", 500))
        self.bg_color: Tuple[int, ...] = tuple(bev.get("bg_color", [40, 40, 40]))
        self.grid_color: Tuple[int, ...] = tuple(
            bev.get("grid_color", [80, 80, 80])
        )
        self.camera_color: Tuple[int, ...] = tuple(
            bev.get("camera_color", [255, 200, 0])
        )
        self.ball_color: Tuple[int, ...] = tuple(
            bev.get("ball_color", [0, 255, 0])
        )
        self.trail_max: int = int(bev.get("trail_max", 500))

        # Internal trail buffers  (world_x, world_z)
        self._camera_trail: List[Tuple[float, float]] = []
        self._ball_trail: List[Tuple[float, float]] = []

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def update(
        self,
        cam_world_pos: np.ndarray,
        ball_world_pos: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record new positions.

        Args:
            cam_world_pos:  3-element array [x, y, z] in world frame.
            ball_world_pos: Dict with ``x_m, z_m`` in world frame, or None.
        """
        self._camera_trail.append(
            (float(cam_world_pos[0]), float(cam_world_pos[2]))
        )
        if ball_world_pos is not None:
            self._ball_trail.append(
                (float(ball_world_pos["x_m"]), float(ball_world_pos["z_m"]))
            )

        # Trim to max length
        if len(self._camera_trail) > self.trail_max:
            self._camera_trail = self._camera_trail[-self.trail_max :]
        if len(self._ball_trail) > self.trail_max:
            self._ball_trail = self._ball_trail[-self.trail_max :]

    def render(self) -> np.ndarray:
        """
        Render the current state of the BEV map.

        Returns:
            BGR image of shape ``(map_size, map_size, 3)``.
        """
        s = self.map_size
        img = np.full((s, s, 3), self.bg_color, dtype=np.uint8)

        all_pts = self._camera_trail + self._ball_trail
        if not all_pts:
            self._draw_title(img)
            self._draw_waiting(img)
            return img

        # ── compute world bounds ──
        xs = [p[0] for p in all_pts]
        zs = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        # Ensure minimum range (at least 4 m in each direction)
        range_x = max(max_x - min_x, 4.0)
        range_z = max(max_z - min_z, 4.0)
        mid_x = (min_x + max_x) / 2.0
        mid_z = (min_z + max_z) / 2.0

        # Add 20 % margin
        range_x *= 1.4
        range_z *= 1.4

        # Drawable area (leave space for title at top, legend at bottom)
        margin_top = 45
        margin_bot = 55
        margin_lr = 25
        draw_w = s - 2 * margin_lr
        draw_h = s - margin_top - margin_bot

        # Uniform scale (pixels per metre)
        ppm = min(draw_w / range_x, draw_h / range_z)

        # Map-pixel centre
        cx_px = s / 2.0
        cy_px = margin_top + draw_h / 2.0

        def to_px(wx: float, wz: float) -> Tuple[int, int]:
            """World metres → map pixel coords (Z-up maps to pixel-Y-up)."""
            px = int(cx_px + (wx - mid_x) * ppm)
            py = int(cy_px - (wz - mid_z) * ppm)  # invert: Z↑ → pixel↑
            return (px, py)

        # ── grid ──
        self._draw_grid(
            img, mid_x, mid_z, range_x, range_z, ppm, to_px, draw_w, draw_h
        )

        # ── camera trail ──
        n_cam = len(self._camera_trail)
        if n_cam >= 2:
            for i in range(n_cam - 1):
                alpha = 0.15 + 0.85 * (i / max(n_cam - 2, 1))
                col = tuple(int(c * alpha) for c in self.camera_color)
                cv2.line(
                    img,
                    to_px(*self._camera_trail[i]),
                    to_px(*self._camera_trail[i + 1]),
                    col,
                    1,
                    cv2.LINE_AA,
                )

        # ── ball trail ──
        n_ball = len(self._ball_trail)
        if n_ball >= 2:
            for i in range(n_ball - 1):
                alpha = 0.25 + 0.75 * (i / max(n_ball - 2, 1))
                col = tuple(int(c * alpha) for c in self.ball_color)
                th = max(1, int(2 * alpha))
                cv2.line(
                    img,
                    to_px(*self._ball_trail[i]),
                    to_px(*self._ball_trail[i + 1]),
                    col,
                    th,
                    cv2.LINE_AA,
                )

        # ── current camera position (triangle marker) ──
        if self._camera_trail:
            cp = to_px(*self._camera_trail[-1])
            cv2.drawMarker(
                img, cp, self.camera_color,
                cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA,
            )

        # ── current ball position (filled circle with outline) ──
        if self._ball_trail:
            bp = to_px(*self._ball_trail[-1])
            cv2.circle(img, bp, 7, self.ball_color, -1, cv2.LINE_AA)
            cv2.circle(img, bp, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # ── decorations ──
        self._draw_title(img)
        self._draw_legend(img)
        self._draw_scale_bar(img, ppm)

        return img

    # ─────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────────────────────────────

    def _draw_grid(
        self,
        img: np.ndarray,
        mid_x: float,
        mid_z: float,
        range_x: float,
        range_z: float,
        ppm: float,
        to_px: Callable,
        draw_w: int,
        draw_h: int,
    ) -> None:
        """Draw metric grid lines with auto-spacing."""
        # Pick a "nice" spacing
        max_range = max(range_x, range_z)
        if max_range <= 5:
            spacing = 1.0
        elif max_range <= 15:
            spacing = 2.0
        elif max_range <= 40:
            spacing = 5.0
        elif max_range <= 100:
            spacing = 10.0
        else:
            spacing = 20.0

        half_x = range_x / 2.0
        half_z = range_z / 2.0

        # Vertical lines (constant world-X)
        x = math.floor((mid_x - half_x) / spacing) * spacing
        while x <= mid_x + half_x:
            p1 = to_px(x, mid_z - half_z)
            p2 = to_px(x, mid_z + half_z)
            cv2.line(img, p1, p2, self.grid_color, 1)
            x += spacing

        # Horizontal lines (constant world-Z)
        z = math.floor((mid_z - half_z) / spacing) * spacing
        while z <= mid_z + half_z:
            p1 = to_px(mid_x - half_x, z)
            p2 = to_px(mid_x + half_x, z)
            cv2.line(img, p1, p2, self.grid_color, 1)
            z += spacing

    @staticmethod
    def _draw_title(img: np.ndarray) -> None:
        cv2.putText(
            img, "2D Top-View Map", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_waiting(img: np.ndarray) -> None:
        h = img.shape[0]
        cv2.putText(
            img, "Waiting for data...", (10, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA,
        )

    def _draw_legend(self, img: np.ndarray) -> None:
        s = self.map_size
        y = s - 18
        # Ball
        cv2.circle(img, (18, y), 5, self.ball_color, -1, cv2.LINE_AA)
        cv2.putText(
            img, "Ball", (28, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )
        # Camera
        cv2.drawMarker(
            img, (95, y), self.camera_color,
            cv2.MARKER_TRIANGLE_UP, 10, 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, "Camera", (110, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    def _draw_scale_bar(self, img: np.ndarray, ppm: float) -> None:
        """Draw a scale bar in the bottom-right corner."""
        s = self.map_size
        # Choose bar length ≈ 1 m, 2 m, 5 m, …
        for bar_m in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            bar_px = int(bar_m * ppm)
            if 40 <= bar_px <= s // 3:
                break
        else:
            # Fall back to whatever 1 m gives
            bar_m = 1.0
            bar_px = max(20, int(ppm))

        y = s - 42
        x2 = s - 20
        x1 = max(20, x2 - bar_px)
        actual_bar_px = x2 - x1

        cv2.line(img, (x1, y), (x2, y), (200, 200, 200), 2)
        cv2.line(img, (x1, y - 5), (x1, y + 5), (200, 200, 200), 1)
        cv2.line(img, (x2, y - 5), (x2, y + 5), (200, 200, 200), 1)

        if actual_bar_px > 0 and ppm > 0:
            actual_m = actual_bar_px / ppm
            label = f"{actual_m:.0f} m" if actual_m >= 1 else f"{actual_m:.1f} m"
        else:
            label = f"{bar_m:.0f} m"
        cv2.putText(
            img, label, (x1, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
        )


# ═══════════════════════════════════════════════════════════════════════
# SPLIT-VIEW COMPOSER
# ═══════════════════════════════════════════════════════════════════════

def create_split_frame(
    video_frame: np.ndarray,
    map_frame: np.ndarray,
    scale_factor: float = 0.75,
) -> np.ndarray:
    """
    Create a side-by-side split-view frame: [video | divider | BEV map].

    Both panels are scaled so that the height equals
    ``video_height x scale_factor``.  The BEV map is scaled uniformly
    to match that height.

    Args:
        video_frame: Annotated BGR video frame (original resolution).
        map_frame:   BEV map BGR image (square).
        scale_factor: Scaling applied to the video height (0 < s ≤ 1).

    Returns:
        Combined BGR image.
    """
    vh, vw = video_frame.shape[:2]
    target_h = int(vh * scale_factor)

    # Scale video panel
    target_vw = int(vw * scale_factor)
    video_scaled = cv2.resize(video_frame, (target_vw, target_h))

    # Scale map panel to same height (keep square aspect)
    mh, mw = map_frame.shape[:2]
    m_scale = target_h / mh
    target_mw = int(mw * m_scale)
    map_scaled = cv2.resize(map_frame, (target_mw, target_h))

    # Thin divider
    divider = np.full((target_h, 2, 3), (100, 100, 100), dtype=np.uint8)

    return np.hstack([video_scaled, divider, map_scaled])
