"""
=============================================================================
Detection Utilities
=============================================================================
Utilities for processing and formatting object detection results.

Usage:
    from tools.detection_utils import extract_detection_data, format_frame_detections
    
    detection = extract_detection_data(box, class_names)
    frame_data = format_frame_detections(frame_idx, fps, detections)
=============================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional


def extract_detection_data(
    box: Any,
    class_names: Dict[int, str]
) -> Dict[str, Any]:
    """
    Extract detection data from a YOLO box result.
    
    Converts raw YOLO detection box to a structured dictionary with
    all relevant information.
    
    Args:
        box: YOLO detection box object (from results.boxes)
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Dictionary with detection data:
            - class_id: Integer class ID
            - class_name: String class name
            - confidence: Float confidence score
            - bbox_xyxy: [x1, y1, x2, y2] coordinates
            - bbox_xywh: [center_x, center_y, width, height]
            - center_px: [center_x, center_y] in pixels
    """
    # Extract raw values from YOLO box
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0].cpu().numpy())
    cls_id = int(box.cls[0].cpu().numpy())
    cls_name = class_names.get(cls_id, f"class_{cls_id}")
    
    # Calculate center and dimensions
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    return {
        "class_id": cls_id,
        "class_name": cls_name,
        "confidence": round(conf, 4),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "bbox_xywh": [round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)],
        "center_px": [round(cx, 2), round(cy, 2)]
    }


def format_frame_detections(
    frame_idx: int,
    fps: int,
    detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format all detections for a single frame.
    
    Creates a structured dictionary containing frame metadata and all
    detections, with special handling for ball detection.
    
    Args:
        frame_idx: Frame index (0-based)
        fps: Video frames per second
        detections: List of detection dictionaries from extract_detection_data
        
    Returns:
        Dictionary with frame data:
            - frame_idx: Frame index
            - timestamp_sec: Time in seconds
            - detections: List of all detections
            - ball_detected: Whether ball was found
            - ball_center: [x, y] if ball detected, else None
            - ball_bbox: [x1, y1, x2, y2] if ball detected, else None
            - ball_confidence: Float if ball detected, else None
    """
    frame_data = {
        "frame_idx": frame_idx,
        "timestamp_sec": round(frame_idx / fps, 4) if fps > 0 else 0.0,
        "detections": detections,
        "ball_detected": False,
        "ball_center": None,
        "ball_bbox": None,
        "ball_confidence": None
    }
    
    # Find ball detection (use highest confidence if multiple)
    ball_detections = [d for d in detections if d["class_name"] == "ball"]
    
    if ball_detections:
        # Get highest confidence ball detection
        best_ball = max(ball_detections, key=lambda d: d["confidence"])
        
        frame_data["ball_detected"] = True
        frame_data["ball_center"] = best_ball["center_px"]
        frame_data["ball_bbox"] = best_ball["bbox_xyxy"]
        frame_data["ball_confidence"] = best_ball["confidence"]
    
    return frame_data


def filter_detections_by_class(
    detections: List[Dict[str, Any]],
    include_classes: Optional[List[str]] = None,
    exclude_classes: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter detections by class name.
    
    Args:
        detections: List of detection dictionaries
        include_classes: Only include these classes (if specified)
        exclude_classes: Exclude these classes (if specified)
        
    Returns:
        Filtered list of detections
    """
    result = detections
    
    if include_classes is not None:
        result = [d for d in result if d["class_name"] in include_classes]
    
    if exclude_classes is not None:
        result = [d for d in result if d["class_name"] not in exclude_classes]
    
    return result


def calculate_detection_stats(
    all_frame_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistics from all frame detections.
    
    Args:
        all_frame_data: List of frame detection dictionaries
        
    Returns:
        Dictionary with statistics:
            - total_frames: Total number of frames
            - ball_detected_frames: Frames where ball was detected
            - ball_detection_rate: Percentage of frames with ball
            - avg_ball_confidence: Average ball confidence when detected
            - total_detections: Total detections across all frames
    """
    total_frames = len(all_frame_data)
    
    if total_frames == 0:
        return {
            "total_frames": 0,
            "ball_detected_frames": 0,
            "ball_detection_rate": 0.0,
            "avg_ball_confidence": 0.0,
            "total_detections": 0
        }
    
    ball_frames = [f for f in all_frame_data if f.get("ball_detected", False)]
    ball_confidences = [
        f["ball_confidence"] 
        for f in ball_frames 
        if f.get("ball_confidence") is not None
    ]
    
    total_detections = sum(
        len(f.get("detections", [])) 
        for f in all_frame_data
    )
    
    return {
        "total_frames": total_frames,
        "ball_detected_frames": len(ball_frames),
        "ball_detection_rate": (len(ball_frames) / total_frames) * 100,
        "avg_ball_confidence": (
            sum(ball_confidences) / len(ball_confidences) 
            if ball_confidences else 0.0
        ),
        "total_detections": total_detections
    }


def get_ball_trajectory(
    all_frame_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract ball trajectory from frame detections.
    
    Returns only frames where ball was detected, useful for
    trajectory tracking and visualization.
    
    Args:
        all_frame_data: List of frame detection dictionaries
        
    Returns:
        List of trajectory points with frame_idx, timestamp, center, bbox
   """
    trajectory = []
    
    for frame_data in all_frame_data:
        if frame_data.get("ball_detected", False):
            trajectory.append({
                "frame_idx": frame_data["frame_idx"],
                "timestamp_sec": frame_data["timestamp_sec"],
                "center": frame_data["ball_center"],
                "bbox": frame_data["ball_bbox"],
                "confidence": frame_data["ball_confidence"]
            })
    
    return trajectory
