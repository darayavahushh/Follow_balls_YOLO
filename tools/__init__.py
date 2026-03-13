"""
Tools Package
=============
Utility modules for the Football Detection & Tracking project.

Modules:
    - config_loader: Configuration file loading and path resolution
    - visualization: Drawing bounding boxes, labels, and overlays
    - video_io: Video capture and writer utilities
    - detection_utils: Detection data processing and formatting
    - logging_utils: Logging configuration and utilities
    - run_manager: Experiment run management
    - depth_estimation: 3D position estimation from pinhole camera model
    - trajectory: Trajectory tracking and trail visualization
"""

from tools.config_loader import load_config, resolve_paths, get_nested_config
from tools.visualization import (
    draw_bbox, 
    draw_label, 
    draw_frame_overlay, 
    get_class_color,
    draw_detection
)
from tools.video_io import VideoReader, VideoWriter, get_video_properties, VideoProperties
from tools.detection_utils import (
    extract_detection_data,
    format_frame_detections,
    filter_detections_by_class,
    calculate_detection_stats,
    get_ball_trajectory
)
from tools.logging_utils import (
    setup_logger,
    get_logger,
    log_summary,
    ProgressLogger,
    capture_output,
    log_function_call
)
from tools.run_manager import RunManager, RunInfo
from tools.depth_estimation import DepthEstimator
from tools.trajectory import TrajectoryTracker

__all__ = [
    # Config
    "load_config",
    "resolve_paths",
    "get_nested_config",
    # Visualization
    "draw_bbox",
    "draw_label", 
    "draw_frame_overlay",
    "get_class_color",
    "draw_detection",
    # Video I/O
    "VideoReader",
    "VideoWriter",
    "get_video_properties",
    "VideoProperties",
    # Detection utilities
    "extract_detection_data",
    "format_frame_detections",
    "filter_detections_by_class",
    "calculate_detection_stats",
    "get_ball_trajectory",
    # Logging
    "setup_logger",
    "get_logger",
    "log_summary",
    "ProgressLogger",
    "capture_output",
    "log_function_call",
    # Run management
    "RunManager",
    "RunInfo",
    # Depth estimation
    "DepthEstimator",
    # Trajectory
    "TrajectoryTracker",
]
