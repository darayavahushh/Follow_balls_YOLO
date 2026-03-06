"""
=============================================================================
Visualization Utilities
=============================================================================
Drawing functions for bounding boxes, labels, and frame overlays.

Usage:
    from tools.visualization import draw_bbox, draw_label, draw_frame_overlay
    
    frame = draw_bbox(frame, x1, y1, x2, y2, color, thickness)
    frame = draw_label(frame, "ball 0.95", x1, y1, color)
=============================================================================
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


def get_class_color(class_name: str, config: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Get BGR color for a class from configuration.
    
    Args:
        class_name: Name of the class (e.g., 'ball', 'person')
        config: Configuration dictionary containing visualization settings
        
    Returns:
        BGR color tuple (Blue, Green, Red) with values 0-255
    """
    colors = config.get('visualization', {}).get('colors', {})
    default_color = colors.get('default', [128, 128, 128])
    color = colors.get(class_name, default_color)
    
    return tuple(color)


def draw_bbox(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box on the frame.
    
    Args:
        frame: Input image (BGR format, modified in place)
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate
        color: BGR color tuple
        thickness: Line thickness in pixels
        
    Returns:
        Frame with bounding box drawn
    """
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness
    )
    
    return frame


def draw_label(
    frame: np.ndarray,
    label: str,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    font_scale: float = 0.6,
    font_thickness: int = 2,
    padding: int = 5
) -> np.ndarray:
    """
    Draw a label with background above a bounding box.
    
    The label is drawn with a filled background rectangle for better visibility.
    
    Args:
        frame: Input image (BGR format, modified in place)
        label: Text to display (e.g., "ball 0.95")
        x: Left x coordinate (same as bbox x1)
        y: Top y coordinate (same as bbox y1)
        color: BGR color tuple for background
        font_scale: Font scale factor
        font_thickness: Font thickness in pixels
        padding: Padding around text in pixels
        
    Returns:
        Frame with label drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (label_w, label_h), baseline = cv2.getTextSize(
        label, 
        font, 
        font_scale, 
        font_thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (int(x), int(y) - label_h - padding * 2),
        (int(x) + label_w + padding, int(y)),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (int(x) + padding // 2, int(y) - padding),
        font,
        font_scale,
        (255, 255, 255),  # White text
        font_thickness
    )
    
    return frame


def draw_frame_overlay(
    frame: np.ndarray,
    frame_idx: int,
    ball_detected: bool,
    config: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Draw frame information overlay (frame number, ball status, etc.).
    
    Args:
        frame: Input image (BGR format, modified in place)
        frame_idx: Current frame index
        ball_detected: Whether ball was detected in this frame
        config: Configuration dictionary with overlay settings
        extra_info: Optional additional info to display
        
    Returns:
        Frame with overlay drawn
    """
    overlay_config = config.get('visualization', {}).get('overlay', {})
    
    # Build info text parts
    info_parts = []
    
    if overlay_config.get('show_frame_number', True):
        info_parts.append(f"Frame: {frame_idx}")
    
    if overlay_config.get('show_ball_status', True):
        status = "YES" if ball_detected else "NO"
        info_parts.append(f"Ball: {status}")
    
    # Add extra info if provided
    if extra_info:
        for key, value in extra_info.items():
            info_parts.append(f"{key}: {value}")
    
    if not info_parts:
        return frame
    
    # Draw overlay
    info_text = " | ".join(info_parts)
    overlay_color = tuple(overlay_config.get('color', [0, 255, 255]))
    font_scale = overlay_config.get('font_scale', 0.8)
    
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        overlay_color,
        2
    )
    
    return frame


def draw_detection(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str,
    confidence: float,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw complete detection visualization (bbox + label).
    
    Convenience function that combines draw_bbox and draw_label with
    configuration-based styling.
    
    Args:
        frame: Input image (BGR format, modified in place)
        x1, y1, x2, y2: Bounding box coordinates
        class_name: Detected class name
        confidence: Detection confidence score (0-1)
        config: Configuration dictionary
        
    Returns:
        Frame with detection drawn
    """
    vis_config = config.get('visualization', {})
    bbox_config = vis_config.get('bbox', {})
    label_config = vis_config.get('labels', {})
    
    # Get color and thickness
    color = get_class_color(class_name, config)
    
    thickness = (
        bbox_config.get('thickness_ball', 3) 
        if class_name == "ball" 
        else bbox_config.get('thickness_default', 2)
    )
    
    # Draw bounding box
    frame = draw_bbox(frame, x1, y1, x2, y2, color, thickness)
    
    # Build and draw label
    if label_config.get('show_confidence', True):
        label = f"{class_name} {confidence:.2f}"
    else:
        label = class_name
    
    frame = draw_label(
        frame,
        label,
        x1,
        y1,
        color,
        font_scale=label_config.get('font_scale', 0.6),
        font_thickness=label_config.get('font_thickness', 2)
    )
    
    return frame
