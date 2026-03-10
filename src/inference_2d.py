"""
=============================================================================
Ball 2D Detection - Inference Pipeline
=============================================================================
Run trained YOLO26 model on video to detect persons and ball.
Outputs annotated video and detection JSON for downstream tasks (3D, BEV).

Usage:
    python src/inference_2d.py --config configs/config.yaml
    python src/inference_2d.py --config configs/config.yaml --ball-only
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
from ultralytics import YOLO

# Local imports from tools
from tools.config_loader import load_config, resolve_paths
from tools.visualization import draw_detection, draw_frame_overlay
from tools.video_io import VideoReader, VideoWriter
from tools.detection_utils import (
    extract_detection_data,
    format_frame_detections,
    calculate_detection_stats
)
from tools.logging_utils import setup_logger, log_summary, ProgressLogger
from tools.run_manager import RunManager


# =============================================================================
# CONSTANTS
# =============================================================================

OPERATION_NAME = "inference_2d"


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def run_inference(
    config: Dict[str, Any],
    run_manager: RunManager,
    logger,
    ball_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Run 2D object detection on video.
    
    Args:
        config: Configuration dictionary
        run_manager: Run manager instance
        logger: Logger instance
        ball_only: Only detect and annotate ball
    
    Returns:
        List of detection dictionaries per frame
    """
    # Extract settings from config
    model_config = config['model']
    inf_config = config['inference']
    paths = config['paths']
    
    # Get model path from run
    model_path = str(run_manager.get_model_path("best.pt"))
    
    # Validate model exists
    if not Path(model_path).exists():
        logger.error(f"Model weights not found: {model_path}")
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Please ensure training was completed for run: {run_manager.run_id}"
        )
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    logger.debug(f"Model path: {model_path}")
    
    model = YOLO(model_path)
    class_names = model.names
    
    logger.debug(f"Model classes: {class_names}")
    
    # Setup paths
    video_path = paths['input_video']
    
    video_stem = Path(video_path).stem
    output_video_path = str(
        run_manager.results_dir / inf_config['output_patterns']['video'].format(video_name=video_stem)
    )
    output_json_path = str(
        run_manager.results_dir / inf_config['output_patterns']['detections'].format(video_name=video_stem)
    )
    
    logger.debug(f"Input video: {video_path}")
    logger.debug(f"Output video: {output_video_path}")
    logger.debug(f"Output JSON: {output_json_path}")
    
    # Process video
    all_detections: List[Dict[str, Any]] = []
    conf_threshold = model_config['confidence_threshold']
    
    with VideoReader(video_path) as reader:
        # Print video info
        props = reader.properties
        
        logger.info("")
        logger.info("📹 Video Properties:")
        logger.info(f"   Resolution: {props.width}x{props.height}")
        logger.info(f"   FPS: {props.fps}")
        logger.info(f"   Total Frames: {props.total_frames}")
        logger.info(f"   Duration: {props.duration_sec:.2f}s")
        
        logger.debug(f"Confidence threshold: {conf_threshold}")
        logger.debug(f"Ball only mode: {ball_only}")
        
        logger.info("")
        logger.info("🔍 Processing video...")
        
        # Setup progress logger
        progress = ProgressLogger(
            logger, 
            props.total_frames, 
            prefix="Progress",
            console_interval=50,
            file_interval=10
        )
        
        with VideoWriter(
            output_video_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            codec=inf_config.get('video_codec', 'XVID')
        ) as writer:
            
            for frame_idx, frame in enumerate(reader):
                # Run detection
                results = model(frame, conf=conf_threshold, verbose=False)[0]
                
                # Extract detections
                frame_detections = []
                
                for box in results.boxes:
                    detection = extract_detection_data(box, class_names)
                    
                    # Filter if ball_only mode
                    if ball_only and detection["class_name"] != "ball":
                        continue
                    
                    frame_detections.append(detection)
                
                # Format frame data
                frame_data = format_frame_detections(
                    frame_idx, 
                    reader.fps, 
                    frame_detections
                )
                all_detections.append(frame_data)
                
                # Draw detections on frame
                annotated_frame = frame.copy()
                
                for det in frame_detections:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    annotated_frame = draw_detection(
                        annotated_frame,
                        x1, y1, x2, y2,
                        det["class_name"],
                        det["confidence"],
                        config
                    )
                
                # Draw frame overlay
                annotated_frame = draw_frame_overlay(
                    annotated_frame,
                    frame_idx,
                    frame_data["ball_detected"],
                    config
                )
                
                # Write frame
                writer.write(annotated_frame)
                
                # Show preview (if enabled and not in quiet mode)
                if inf_config.get('show_preview', False):
                    cv2.imshow("2D Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.warning("Preview closed by user")
                        break
                
                # Update progress
                progress.update(frame_idx + 1)
    
    # Cleanup preview window
    cv2.destroyAllWindows()
    
    # Calculate stats
    stats = calculate_detection_stats(all_detections)
    
    logger.debug(f"Total detections: {stats['total_detections']}")
    logger.debug(f"Ball detection rate: {stats['ball_detection_rate']:.1f}%")
    
    # Save detections JSON
    output_data = {
        "run_id": run_manager.run_id,
        "video_info": props.to_dict(),
        "model_info": {
            "weights": model_path,
            "confidence_threshold": conf_threshold,
            "classes": {int(k): v for k, v in class_names.items()}
        },
        "statistics": stats,
        "frames": all_detections
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    logger.debug(f"Detections saved to: {output_json_path}")
    
    # Print summary
    logger.info("")
    logger.info("✅ Processing complete!")
    logger.info(f"      Output video: {output_video_path}")
    logger.info(f"      Detections JSON: {output_json_path}")
    logger.info(f"      Ball detected in {stats['ball_detected_frames']}/{stats['total_frames']} frames ({stats['ball_detection_rate']:.1f}%)")
    
    return all_detections


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main inference execution."""
    
    parser = argparse.ArgumentParser(
        description="Run 2D ball detection on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to use (e.g., run_001). Use existing run with trained model."
    )
    parser.add_argument(
        "--ball-only",
        action="store_true",
        help="Only detect and annotate the ball"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable real-time preview window"
    )
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_paths(config, PROJECT_ROOT)
    
    # Apply CLI overrides
    if args.no_preview:
        config['inference']['show_preview'] = False
    
    # Load existing run
    run_manager = RunManager.load_run(args.run_id, PROJECT_ROOT)
    run_manager.add_operation(OPERATION_NAME)
    
    # Setup logger
    log_path = run_manager.get_log_path(OPERATION_NAME)
    logger = setup_logger(OPERATION_NAME, log_path, config)
    
    # Log run info
    logger.info("")
    logger.info("=" * 60)
    logger.info("⚽ BALL 2D DETECTION - INFERENCE")
    logger.info("=" * 60)
    logger.info(f"   Run ID: {run_manager.run_id}")
    logger.info(f"   Config: {args.config}")
    logger.info(f"   Log: {log_path}")
    
    logger.debug(f"Project root: {PROJECT_ROOT}")
    logger.debug(f"Run directory: {run_manager.run_dir}")
    logger.debug(f"Ball only: {args.ball_only}")
    
    try:
        # Run inference
        all_detections = run_inference(
            config,
            run_manager,
            logger,
            ball_only=args.ball_only
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get stats for summary
        stats = calculate_detection_stats(all_detections)
        
        # Log summary
        log_summary(logger, "SUCCESS", duration, {
            "frames_processed": stats['total_frames'],
            "ball_detection_rate": f"{stats['ball_detection_rate']:.1f}%",
            "total_detections": stats['total_detections']
        })
        
        # Final console output
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ INFERENCE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Run ID: {run_manager.run_id}")
        logger.info(f"   Results: {run_manager.results_dir}")
        logger.info(f"   Log: {log_path}")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Inference failed: {e}")
        log_summary(logger, "FAILED", duration, {"error": str(e)})
        raise


if __name__ == "__main__":
    main()