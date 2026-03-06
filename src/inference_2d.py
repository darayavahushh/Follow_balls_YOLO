"""
=============================================================================
Football 2D Detection - Inference Pipeline
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
    filter_detections_by_class,
    calculate_detection_stats
)


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def run_inference(
    config: Dict[str, Any], 
    model_path: Optional[str] = None, 
    ball_only: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Run 2D object detection on video.
    
    Detects 'person' and 'ball' in each frame, draws bounding boxes,
    and saves results for downstream processing (3D detection, trajectory).
    
    Args:
        config: Configuration dictionary
        model_path: Override model path from config
        ball_only: Override ball_only setting from config
    
    Returns:
        List of detection dictionaries per frame
    """
    # Extract settings from config
    model_config = config['model']
    inf_config = config['inference']
    paths = config['paths']
    
    # Resolve model path
    if model_path is None:
        model_path = str(
            Path(paths['models_dir']) / "train_run" / "weights" / "best.pt"
        )
    
    # Validate model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Please train a model first with: python src/train_2d.py --config configs/config.yaml"
        )
    
    # Resolve ball_only setting
    if ball_only is None:
        ball_only = inf_config.get('ball_only', False)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    
    # Setup paths
    video_path = paths['input_video']
    results_dir = Path(paths['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    video_stem = Path(video_path).stem
    output_video_path = str(
        results_dir / inf_config['output_patterns']['video'].format(video_name=video_stem)
    )
    output_json_path = str(
        results_dir / inf_config['output_patterns']['detections'].format(video_name=video_stem)
    )
    
    # Process video
    all_detections: List[Dict[str, Any]] = []
    conf_threshold = model_config['confidence_threshold']
    
    with VideoReader(video_path) as reader:
        # Print video info
        props = reader.properties
        print(f"\n📹 Video Properties:")
        print(f"   Resolution: {props.width}x{props.height}")
        print(f"   FPS: {props.fps}")
        print(f"   Total Frames: {props.total_frames}")
        print(f"   Duration: {props.duration_sec:.2f}s")
        
        print(f"\n Processing video...")
        
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
                
                # Show preview
                if inf_config.get('show_preview', True):
                    cv2.imshow("2D Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️ Preview closed by user")
                        break
                
                # Progress update
                if (frame_idx + 1) % 50 == 0:
                    progress = ((frame_idx + 1) / reader.total_frames) * 100
                    print(f"   Progress: {frame_idx + 1}/{reader.total_frames} ({progress:.1f}%)")
    
    # Cleanup preview window
    cv2.destroyAllWindows()
    
    # Calculate and print stats
    stats = calculate_detection_stats(all_detections)
    
    # Save detections JSON
    output_data = {
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
    
    # Print summary
    print(f"\n✅ Processing complete!")
    print(f"    Output video: {output_video_path}")
    print(f"    Detections JSON: {output_json_path}")
    print(f"    Ball detected in {stats['ball_detected_frames']}/{stats['total_frames']} frames ({stats['ball_detection_rate']:.1f}%)")
    
    return all_detections


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main inference execution."""
    
    parser = argparse.ArgumentParser(
        description="Run 2D football detection on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model path from config"
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
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_paths(config, PROJECT_ROOT)
    
    # Apply CLI overrides
    if args.no_preview:
        config['inference']['show_preview'] = False
    
    ball_only = args.ball_only if args.ball_only else None
    
    print("\n" + "=" * 60)
    print("⚽ BALL 2D DETECTION - INFERENCE")
    print("=" * 60)
    print(f"   Config: {args.config}")
    
    run_inference(config, model_path=args.model, ball_only=ball_only)


if __name__ == "__main__":
    main()
