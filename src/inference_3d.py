"""
=============================================================================
Ball 3D Detection & Trajectory Tracking - Inference Pipeline
=============================================================================
Runs the trained YOLO26 model on a video, then for every frame:

  1.  Detects the ball (2D bounding-box).
  2.  Estimates its 3D position (X, Y, Z) and distance using the pinhole
      camera model + known football diameter.
  3.  Maintains a trajectory trail and draws it on the frame.

Outputs:
  • Annotated video  - bounding boxes + 3D info overlay + trajectory trail.
  • Excel (.xlsx)    - per-frame table with columns:
        frame, time_s, x_m, y_m, z_m, distance_m, cx_px, cy_px

Usage:
    python src/inference_3d.py --config configs/config.yaml --run-id run_001
    python src/inference_3d.py --config configs/config.yaml --run-id run_001 --no-preview
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO

# Local imports from tools
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


# =============================================================================
# CONSTANTS
# =============================================================================

OPERATION_NAME = "inference_3d"


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def save_results_excel(
    records: List[Dict[str, Any]],
    output_path: str,
    logger,
) -> None:
    """
    Save 3D detection results to an Excel (.xlsx) file.

    Each row corresponds to a frame where the ball was detected.

    Args:
        records:     List of dicts with keys frame, time_s, x_m, y_m, z_m,
                     distance_m, cx_px, cy_px, confidence.
        output_path: Destination .xlsx path.
        logger:      Logger instance.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning(
            "openpyxl not installed – falling back to CSV export.  "
            "Install with: pip install openpyxl"
        )
        _save_results_csv(records, output_path.replace(".xlsx", ".csv"), logger)
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "3D Ball Positions"

    # Header
    headers = [
        "Frame", "Time (s)",
        "X (m)", "Y (m)", "Z (m)", "Distance (m)",
        "Center X (px)", "Center Y (px)", "Confidence",
    ]
    ws.append(headers)

    # Style header
    from openpyxl.styles import Font, Alignment
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    # Data rows
    for r in records:
        ws.append([
            r["frame"],
            round(r["time_s"], 4),
            round(r["x_m"], 4),
            round(r["y_m"], 4),
            round(r["z_m"], 4),
            round(r["distance_m"], 4),
            round(r["cx_px"], 2),
            round(r["cy_px"], 2),
            round(r["confidence"], 4),
        ])

    # Auto-width columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3

    wb.save(output_path)
    logger.info(f"      Excel: {output_path}")


def _save_results_csv(
    records: List[Dict[str, Any]],
    output_path: str,
    logger,
) -> None:
    """Fallback CSV export when openpyxl is not available."""
    import csv

    fieldnames = [
        "frame", "time_s",
        "x_m", "y_m", "z_m", "distance_m",
        "cx_px", "cy_px", "confidence",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"      CSV (fallback): {output_path}")


# =============================================================================
# 3D INFO OVERLAY
# =============================================================================

def draw_3d_overlay(
    frame: np.ndarray,
    pos_3d: Optional[Dict[str, float]],
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Draw the estimated 3D position + distance text on the frame.

    Placed just below the standard frame-overlay line.

    Args:
        frame:  BGR image (modified in-place).
        pos_3d: Result dict from DepthEstimator.estimate(), or None.
        config: Project config.

    Returns:
        Annotated frame.
    """
    if pos_3d is None:
        text = "3D: N/A"
    else:
        text = DepthEstimator.format_position_str(pos_3d)

    overlay_cfg = config.get("visualization", {}).get("overlay", {})
    color = tuple(overlay_cfg.get("color", [0, 255, 255]))
    font_scale = overlay_cfg.get("font_scale", 0.8)

    cv2.putText(
        frame,
        text,
        (10, 60),  # second line, below the standard overlay at y=30
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.7,
        color,
        2,
    )

    return frame


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def run_inference_3d(
    config: Dict[str, Any],
    run_manager: RunManager,
    logger,
) -> List[Dict[str, Any]]:
    """
    Run combined 2D detection + 3D estimation + trajectory tracking.

    Args:
        config:       Configuration dictionary.
        run_manager:  Run manager instance.
        logger:       Logger instance.

    Returns:
        List of per-frame result dictionaries.
    """
    # ── settings ────────────────────────────────────────────────────
    model_config = config["model"]
    inf_config = config["inference"]
    traj_config = config.get("trajectory", {})
    paths = config["paths"]
    conf_threshold = model_config["confidence_threshold"]

    # ── model ───────────────────────────────────────────────────────
    model_path = str(run_manager.get_model_path("best.pt"))

    if not Path(model_path).exists():
        logger.error(f"Model weights not found: {model_path}")
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Please ensure training was completed for run: {run_manager.run_id}"
        )

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    logger.debug(f"Model classes: {class_names}")

    # ── 3D estimator & trajectory tracker ───────────────────────────
    estimator = DepthEstimator(config)
    tracker = TrajectoryTracker(config)

    logger.debug(f"Ball real diameter: {estimator.ball_diameter_m:.3f} m")
    logger.debug(f"Trajectory trail length: {tracker.trail_length}")

    # ── paths ───────────────────────────────────────────────────────
    video_path = paths["input_video"]
    video_stem = Path(video_path).stem

    output_patterns = inf_config.get("output_patterns", {})
    output_video_path = str(
        run_manager.results_dir
        / output_patterns.get("video_3d", "{video_name}_3d_trajectory.avi").format(
            video_name=video_stem
        )
    )
    output_excel_path = str(
        run_manager.results_dir
        / output_patterns.get("excel_3d", "{video_name}_3d_positions.xlsx").format(
            video_name=video_stem
        )
    )
    output_json_path = str(
        run_manager.results_dir
        / output_patterns.get("detections_3d", "{video_name}_3d_detections.json").format(
            video_name=video_stem
        )
    )

    logger.debug(f"Input video:  {video_path}")
    logger.debug(f"Output video: {output_video_path}")
    logger.debug(f"Output Excel: {output_excel_path}")
    logger.debug(f"Output JSON:  {output_json_path}")

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
        logger.info("🔍 Processing video (3D + trajectory)...")

        progress = ProgressLogger(
            logger,
            props.total_frames,
            prefix="Progress",
            console_interval=50,
            file_interval=10,
        )

        with VideoWriter(
            output_video_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            codec=inf_config.get("video_codec", "XVID"),
        ) as writer:

            for frame_idx, frame in enumerate(reader):
                # ── 2D detection ────────────────────────────────────
                results = model(frame, conf=conf_threshold, verbose=False)[0]

                frame_detections = []
                for box in results.boxes:
                    detection = extract_detection_data(box, class_names)
                    frame_detections.append(detection)

                frame_data = format_frame_detections(
                    frame_idx, reader.fps, frame_detections
                )

                # ── 3D estimation (ball only) ───────────────────────
                pos_3d: Optional[Dict[str, float]] = None

                if frame_data["ball_detected"]:
                    pos_3d = estimator.estimate(
                        frame_data["ball_bbox"],
                        (reader.height, reader.width),
                    )
                    frame_data["pos_3d"] = pos_3d

                    # Record for Excel
                    excel_records.append(
                        {
                            "frame": frame_idx,
                            "time_s": frame_data["timestamp_sec"],
                            "x_m": pos_3d["x_m"],
                            "y_m": pos_3d["y_m"],
                            "z_m": pos_3d["z_m"],
                            "distance_m": pos_3d["distance_m"],
                            "cx_px": frame_data["ball_center"][0],
                            "cy_px": frame_data["ball_center"][1],
                            "confidence": frame_data["ball_confidence"],
                        }
                    )
                else:
                    frame_data["pos_3d"] = None

                all_frame_data.append(frame_data)

                # ── trajectory update ───────────────────────────────
                tracker.update(
                    frame_idx,
                    frame_data["ball_center"],
                    pos_3d,
                )

                # ── draw annotated frame ────────────────────────────
                annotated = frame.copy()

                # Bounding boxes (all detections)
                for det in frame_detections:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    annotated = draw_detection(
                        annotated, x1, y1, x2, y2,
                        det["class_name"], det["confidence"], config,
                    )

                # Standard overlay (frame number + ball status)
                annotated = draw_frame_overlay(
                    annotated, frame_idx, frame_data["ball_detected"], config,
                )

                # 3D info overlay
                annotated = draw_3d_overlay(annotated, pos_3d, config)

                # Trajectory trail
                annotated = tracker.draw(annotated, config)

                writer.write(annotated)

                # Preview
                if inf_config.get("show_preview", False):
                    cv2.imshow("3D Detection + Trajectory", annotated)
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
    save_results_excel(excel_records, output_excel_path, logger)

    # ── save JSON ───────────────────────────────────────────────────
    # Make pos_3d JSON-safe (NaN → null)
    def _sanitize(obj):
        """Replace NaN/Inf with None for JSON compatibility."""
        if isinstance(obj, float) and (obj != obj or obj == float("inf")):
            return None
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
            "fx": estimator.fx,
            "fy": estimator.fy,
            "cx": estimator.cx,
            "cy": estimator.cy,
            "ball_diameter_m": estimator.ball_diameter_m,
        },
        "statistics": stats,
        "frames": _sanitize(all_frame_data),
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.debug(f"JSON saved to: {output_json_path}")

    # ── summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("✅ Processing complete!")
    logger.info(f"      Output video: {output_video_path}")
    logger.info(f"      Excel:        {output_excel_path}")
    logger.info(f"      JSON:         {output_json_path}")
    logger.info(
        f"      Ball detected in {stats['ball_detected_frames']}/"
        f"{stats['total_frames']} frames "
        f"({stats['ball_detection_rate']:.1f}%)"
    )
    logger.info(
        f"      3D records:   {len(excel_records)} rows written to Excel"
    )

    return all_frame_data


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main entry point for 3D detection + trajectory inference."""

    parser = argparse.ArgumentParser(
        description="Run 3D ball detection & trajectory tracking on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to use (e.g., run_001). Must contain trained weights.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable real-time preview window",
    )

    args = parser.parse_args()

    # Record start time
    start_time = time.time()

    # Load and resolve config
    config = load_config(args.config)
    config = resolve_paths(config, PROJECT_ROOT)

    # Apply CLI overrides
    if args.no_preview:
        config["inference"]["show_preview"] = False

    # Load existing run
    run_manager = RunManager.load_run(args.run_id, PROJECT_ROOT)
    run_manager.add_operation(OPERATION_NAME)

    # Setup logger
    log_path = run_manager.get_log_path(OPERATION_NAME)
    logger = setup_logger(OPERATION_NAME, log_path, config)

    # Log run info
    logger.info("")
    logger.info("=" * 60)
    logger.info("⚽ BALL 3D DETECTION & TRAJECTORY TRACKING")
    logger.info("=" * 60)
    logger.info(f"   Run ID:  {run_manager.run_id}")
    logger.info(f"   Config:  {args.config}")
    logger.info(f"   Log:     {log_path}")

    logger.debug(f"Project root: {PROJECT_ROOT}")
    logger.debug(f"Run directory: {run_manager.run_dir}")

    try:
        all_frame_data = run_inference_3d(config, run_manager, logger)

        duration = time.time() - start_time
        stats = calculate_detection_stats(all_frame_data)

        log_summary(logger, "SUCCESS", duration, {
            "frames_processed": stats["total_frames"],
            "ball_detection_rate": f"{stats['ball_detection_rate']:.1f}%",
            "total_detections": stats["total_detections"],
        })

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 3D DETECTION & TRAJECTORY COMPLETE")
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
