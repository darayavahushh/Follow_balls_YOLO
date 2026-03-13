"""
=============================================================================
Ball 2D Top-View Map (BEV) - Inference Pipeline
=============================================================================
Produces a split-view video:

    Left  panel  - original video with bounding boxes, 3D overlay, and
                   trajectory trail (same as inference_3d).
    Right panel  - 2D top-view map showing the ball and camera positions
                   in a fixed world frame that is **not affected by camera
                   movement or shaking**.

The ball's world-frame position is obtained by:
    1.  Detecting the ball (YOLO) → 2D bounding box.
    2.  Estimating depth via the known-ball-diameter + pinhole model → 3D
        position in camera frame (X, Y, Z).
    3.  Estimating frame-to-frame camera ego-motion via sparse optical
        flow + Essential matrix decomposition.
    4.  Accumulating the camera pose to transform the ball from camera
        frame to a fixed world frame.

Outputs:
    • Split-view video (.avi)  - [annotated video | 2D top-view map]
    • Excel (.xlsx)            - per-frame ball world positions
    • JSON                     - full detection + 3D + world data

Usage:
    python src/inference_bev.py --config configs/config.yaml --run-id run_001
    python src/inference_bev.py --config configs/config.yaml --run-id run_001 --no-preview
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from tools.config_loader import load_config, resolve_paths
from tools.visualization import draw_detection, draw_frame_overlay
from tools.video_io import VideoReader, VideoWriter
from tools.detection_utils import (
    extract_detection_data,
    format_frame_detections,
    calculate_detection_stats,
)
from tools.logging_utils import setup_logger, log_summary, ProgressLogger
from tools.run_manager import RunManager
from tools.depth_estimation import DepthEstimator
from tools.trajectory import TrajectoryTracker
from tools.bev_map import CameraMotionEstimator, BEVMapRenderer, create_split_frame


# =============================================================================
# CONSTANTS
# =============================================================================

OPERATION_NAME = "inference_bev"


# =============================================================================
# EXCEL EXPORT (reused from inference_3d with world-frame columns)
# =============================================================================

def save_bev_excel(
    records: List[Dict[str, Any]],
    output_path: str,
    logger,
) -> None:
    """
    Save BEV results to Excel.  Each row is a frame where the ball was
    detected.  Columns include both camera-frame and world-frame
    positions.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning(
            "openpyxl not installed - falling back to CSV.  "
            "Install with: pip install openpyxl"
        )
        _save_bev_csv(records, output_path.replace(".xlsx", ".csv"), logger)
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BEV Ball Positions"

    headers = [
        "Frame", "Time (s)",
        "Cam X (m)", "Cam Y (m)", "Cam Z (m)",
        "Ball X cam (m)", "Ball Y cam (m)", "Ball Z cam (m)",
        "Distance (m)",
        "Ball X world (m)", "Ball Y world (m)", "Ball Z world (m)",
        "Confidence",
    ]
    ws.append(headers)

    from openpyxl.styles import Font, Alignment
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for r in records:
        ws.append([
            r["frame"],
            round(r["time_s"], 4),
            round(r["cam_x"], 4),
            round(r["cam_y"], 4),
            round(r["cam_z"], 4),
            round(r["ball_x_cam"], 4),
            round(r["ball_y_cam"], 4),
            round(r["ball_z_cam"], 4),
            round(r["distance_m"], 4),
            round(r["ball_x_world"], 4),
            round(r["ball_y_world"], 4),
            round(r["ball_z_world"], 4),
            round(r["confidence"], 4),
        ])

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3

    wb.save(output_path)
    logger.info(f"      Excel: {output_path}")


def _save_bev_csv(
    records: List[Dict[str, Any]],
    output_path: str,
    logger,
) -> None:
    import csv
    fieldnames = [
        "frame", "time_s",
        "cam_x", "cam_y", "cam_z",
        "ball_x_cam", "ball_y_cam", "ball_z_cam",
        "distance_m",
        "ball_x_world", "ball_y_world", "ball_z_world",
        "confidence",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"      CSV (fallback): {output_path}")


# =============================================================================
# 3D INFO OVERLAY (same as inference_3d)
# =============================================================================

def _draw_3d_overlay(
    frame: np.ndarray,
    pos_3d: Optional[Dict[str, float]],
    config: Dict[str, Any],
) -> np.ndarray:
    if pos_3d is None:
        text = "3D: N/A"
    else:
        text = DepthEstimator.format_position_str(pos_3d)

    overlay_cfg = config.get("visualization", {}).get("overlay", {})
    color = tuple(overlay_cfg.get("color", [0, 255, 255]))
    font_scale = overlay_cfg.get("font_scale", 0.8)

    cv2.putText(
        frame, text, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, 2,
    )
    return frame


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def run_inference_bev(
    config: Dict[str, Any],
    run_manager: RunManager,
    logger,
) -> List[Dict[str, Any]]:
    """
    Run 2D detection + 3D estimation + camera motion + BEV map generation.
    Produces a split-view video and an Excel file with world-frame
    ball positions.
    """
    # ── settings ────────────────────────────────────────────────────
    model_config = config["model"]
    inf_config = config["inference"]
    bev_config = config.get("bev", {})
    paths = config["paths"]
    conf_threshold = model_config["confidence_threshold"]
    split_scale = float(bev_config.get("split_video_scale", 0.75))

    # ── model ───────────────────────────────────────────────────────
    model_path = str(run_manager.get_model_path("best.pt"))
    if not Path(model_path).exists():
        logger.error(f"Model weights not found: {model_path}")
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Ensure training was completed for run: {run_manager.run_id}"
        )

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    logger.debug(f"Model classes: {class_names}")

    # ── modules ─────────────────────────────────────────────────────
    estimator = DepthEstimator(config)
    tracker = TrajectoryTracker(config)
    motion = CameraMotionEstimator(config)
    bev_renderer = BEVMapRenderer(config)

    logger.debug(f"Ball diameter: {estimator.ball_diameter_m:.3f} m")
    logger.debug(f"Trajectory trail: {tracker.trail_length}")
    logger.debug(f"BEV map size: {bev_renderer.map_size}px")
    logger.debug(f"Split scale: {split_scale}")

    # ── output paths ────────────────────────────────────────────────
    video_path = paths["input_video"]
    video_stem = Path(video_path).stem
    patterns = inf_config.get("output_patterns", {})

    output_video = str(
        run_manager.results_dir
        / patterns.get(
            "video_bev", "{video_name}_bev_split.avi"
        ).format(video_name=video_stem)
    )
    output_excel = str(
        run_manager.results_dir
        / patterns.get(
            "excel_bev", "{video_name}_bev_positions.xlsx"
        ).format(video_name=video_stem)
    )
    output_json = str(
        run_manager.results_dir
        / patterns.get(
            "detections_bev", "{video_name}_bev_detections.json"
        ).format(video_name=video_stem)
    )

    logger.debug(f"Input video:  {video_path}")
    logger.debug(f"Output video: {output_video}")
    logger.debug(f"Output Excel: {output_excel}")
    logger.debug(f"Output JSON:  {output_json}")

    # ── process video ───────────────────────────────────────────────
    all_frame_data: List[Dict[str, Any]] = []
    excel_records: List[Dict[str, Any]] = []

    with VideoReader(video_path) as reader:
        props = reader.properties

        logger.info("")
        logger.info("📹 Video Properties:")
        logger.info(f"   Resolution: {props.width}x{props.height}")
        logger.info(f"   FPS: {props.fps}")
        logger.info(f"   Total Frames: {props.total_frames}")
        logger.info(f"   Duration: {props.duration_sec:.2f}s")
        logger.info("")
        logger.info("🔍 Processing video (BEV split-view)...")

        progress = ProgressLogger(
            logger, props.total_frames,
            prefix="Progress", console_interval=50, file_interval=10,
        )

        # Compute split-view output dimensions for the VideoWriter
        target_h = int(props.height * split_scale)
        target_vw = int(props.width * split_scale)
        map_h = bev_renderer.map_size
        target_mw = int(bev_renderer.map_size * (target_h / map_h))
        out_w = target_vw + 2 + target_mw  # +2 for divider

        with VideoWriter(
            output_video,
            fps=reader.fps,
            width=out_w,
            height=target_h,
            codec=inf_config.get("video_codec", "XVID"),
        ) as writer:

            for frame_idx, frame in enumerate(reader):
                # ── 2D detection ────────────────────────────────────
                results = model(frame, conf=conf_threshold, verbose=False)[0]

                frame_detections = []
                for box in results.boxes:
                    det = extract_detection_data(box, class_names)
                    frame_detections.append(det)

                frame_data = format_frame_detections(
                    frame_idx, reader.fps, frame_detections,
                )

                # ── 3D estimation ───────────────────────────────────
                pos_3d: Optional[Dict[str, float]] = None
                ball_depth: Optional[float] = None

                if frame_data["ball_detected"]:
                    pos_3d = estimator.estimate(
                        frame_data["ball_bbox"],
                        (reader.height, reader.width),
                    )
                    ball_depth = pos_3d.get("z_m")
                    frame_data["pos_3d"] = pos_3d
                else:
                    frame_data["pos_3d"] = None

                # ── camera motion ───────────────────────────────────
                motion.update(frame, ball_depth=ball_depth)
                cam_world = motion.get_camera_world_pos()

                # ── ball → world frame ──────────────────────────────
                ball_world: Optional[Dict[str, float]] = None
                if pos_3d is not None:
                    ball_world = motion.transform_to_world(pos_3d)

                frame_data["ball_world"] = ball_world
                frame_data["cam_world"] = {
                    "x_m": float(cam_world[0]),
                    "y_m": float(cam_world[1]),
                    "z_m": float(cam_world[2]),
                }
                all_frame_data.append(frame_data)

                # ── BEV update ──────────────────────────────────────
                bev_renderer.update(cam_world, ball_world)

                # ── trajectory update ───────────────────────────────
                tracker.update(frame_idx, frame_data["ball_center"], pos_3d)

                # ── Excel record ────────────────────────────────────
                if pos_3d is not None and ball_world is not None:
                    excel_records.append({
                        "frame": frame_idx,
                        "time_s": frame_data["timestamp_sec"],
                        "cam_x": cam_world[0],
                        "cam_y": cam_world[1],
                        "cam_z": cam_world[2],
                        "ball_x_cam": pos_3d["x_m"],
                        "ball_y_cam": pos_3d["y_m"],
                        "ball_z_cam": pos_3d["z_m"],
                        "distance_m": pos_3d["distance_m"],
                        "ball_x_world": ball_world["x_m"],
                        "ball_y_world": ball_world["y_m"],
                        "ball_z_world": ball_world["z_m"],
                        "confidence": frame_data["ball_confidence"],
                    })

                # ── annotate left panel ─────────────────────────────
                annotated = frame.copy()

                for det in frame_detections:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    annotated = draw_detection(
                        annotated, x1, y1, x2, y2,
                        det["class_name"], det["confidence"], config,
                    )

                annotated = draw_frame_overlay(
                    annotated, frame_idx,
                    frame_data["ball_detected"], config,
                )
                annotated = _draw_3d_overlay(annotated, pos_3d, config)
                annotated = tracker.draw(annotated, config)

                # ── render BEV map (right panel) ────────────────────
                map_img = bev_renderer.render()

                # ── compose split-view ──────────────────────────────
                split = create_split_frame(annotated, map_img, split_scale)
                writer.write(split)

                # Preview
                if inf_config.get("show_preview", False):
                    cv2.imshow("BEV Split View", split)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.warning("Preview closed by user")
                        break

                progress.update(frame_idx + 1)

    cv2.destroyAllWindows()

    # ── stats ───────────────────────────────────────────────────────
    stats = calculate_detection_stats(all_frame_data)
    logger.debug(f"Total detections: {stats['total_detections']}")
    logger.debug(f"Ball detection rate: {stats['ball_detection_rate']:.1f}%")

    # ── save Excel ──────────────────────────────────────────────────
    save_bev_excel(excel_records, output_excel, logger)

    # ── save JSON ───────────────────────────────────────────────────
    def _sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    output_data = {
        "run_id": run_manager.run_id,
        "video_info": props.to_dict(),
        "model_info": {
            "weights": model_path,
            "confidence_threshold": conf_threshold,
            "classes": {int(k): v for k, v in class_names.items()},
        },
        "camera_info": {
            "fx": estimator.fx, "fy": estimator.fy,
            "cx": estimator.cx, "cy": estimator.cy,
            "ball_diameter_m": estimator.ball_diameter_m,
        },
        "statistics": stats,
        "frames": _sanitize(all_frame_data),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.debug(f"JSON saved to: {output_json}")

    # ── summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("✅ Processing complete!")
    logger.info(f"      Split video: {output_video}")
    logger.info(f"      Excel:       {output_excel}")
    logger.info(f"      JSON:        {output_json}")
    logger.info(
        f"      Ball detected in {stats['ball_detected_frames']}/"
        f"{stats['total_frames']} frames "
        f"({stats['ball_detection_rate']:.1f}%)"
    )
    logger.info(f"      BEV records: {len(excel_records)} rows in Excel")

    return all_frame_data


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main entry point for BEV split-view inference."""

    parser = argparse.ArgumentParser(
        description="Generate 2D top-view map with split-view video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="Run ID to use (e.g., run_001). Must contain trained weights.",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable real-time preview window",
    )

    args = parser.parse_args()
    start_time = time.time()

    # Load and resolve config
    config = load_config(args.config)
    config = resolve_paths(config, PROJECT_ROOT)

    if args.no_preview:
        config["inference"]["show_preview"] = False

    # Load existing run
    run_manager = RunManager.load_run(args.run_id, PROJECT_ROOT)
    run_manager.add_operation(OPERATION_NAME)

    # Setup logger
    log_path = run_manager.get_log_path(OPERATION_NAME)
    logger = setup_logger(OPERATION_NAME, log_path, config)

    logger.info("")
    logger.info("=" * 60)
    logger.info("⚽ BALL 2D TOP-VIEW MAP (BEV) - SPLIT VIEW")
    logger.info("=" * 60)
    logger.info(f"   Run ID:  {run_manager.run_id}")
    logger.info(f"   Config:  {args.config}")
    logger.info(f"   Log:     {log_path}")

    logger.debug(f"Project root: {PROJECT_ROOT}")
    logger.debug(f"Run directory: {run_manager.run_dir}")

    try:
        all_frame_data = run_inference_bev(config, run_manager, logger)

        duration = time.time() - start_time
        stats = calculate_detection_stats(all_frame_data)

        log_summary(logger, "SUCCESS", duration, {
            "frames_processed": stats["total_frames"],
            "ball_detection_rate": f"{stats['ball_detection_rate']:.1f}%",
            "total_detections": stats["total_detections"],
        })

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ BEV SPLIT-VIEW COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Run ID:   {run_manager.run_id}")
        logger.info(f"   Results:  {run_manager.results_dir}")
        logger.info(f"   Log:      {log_path}")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Inference failed: {e}")
        log_summary(logger, "FAILED", duration, {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
