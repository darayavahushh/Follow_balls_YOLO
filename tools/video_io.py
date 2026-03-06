"""
=============================================================================
Video I/O Utilities
=============================================================================
Video reading and writing utilities with context manager support.

Usage:
    from tools.video_io import VideoReader, VideoWriter
    
    with VideoReader("input.avi") as reader:
        for frame in reader:
            process(frame)
=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass


@dataclass
class VideoProperties:
    """Container for video properties."""
    width: int
    height: int
    fps: int
    total_frames: int
    duration_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2)
        }


def get_video_properties(video_path: str) -> VideoProperties:
    """
    Get properties of a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoProperties dataclass with video metadata
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        return VideoProperties(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec
        )
    finally:
        cap.release()


class VideoReader:
    """
    Video reader with context manager and iterator support.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
            
        Raises:
            FileNotFoundError: If video doesn't exist
        """
        self.video_path = video_path
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._properties: Optional[VideoProperties] = None
    
    def __enter__(self) -> "VideoReader":
        """Open video for reading."""
        self._cap = cv2.VideoCapture(self.video_path)
        
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self._properties = get_video_properties(self.video_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames."""
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as context manager")
        
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame.
        
        Returns:
            Tuple of (success, frame)
        """
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as context manager")
        
        return self._cap.read()
    
    @property
    def properties(self) -> VideoProperties:
        """Get video properties."""
        if self._properties is None:
            raise RuntimeError("VideoReader must be used as context manager")
        return self._properties
    
    @property
    def width(self) -> int:
        """Video width in pixels."""
        return self.properties.width
    
    @property
    def height(self) -> int:
        """Video height in pixels."""
        return self.properties.height
    
    @property
    def fps(self) -> int:
        """Video frames per second."""
        return self.properties.fps
    
    @property
    def total_frames(self) -> int:
        """Total number of frames."""
        return self.properties.total_frames
    
    @property
    def duration_sec(self) -> float:
        """Video duration in seconds."""
        return self.properties.duration_sec


class VideoWriter:
    """
    Video writer with context manager support.
    """
    
    def __init__(
        self,
        output_path: str,
        fps: int,
        width: int,
        height: int,
        codec: str = "XVID"
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second
            width: Frame width in pixels
            height: Frame height in pixels
            codec: FourCC codec string (default: XVID)
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count: int = 0
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def __enter__(self) -> "VideoWriter":
        """Open video for writing."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not self._writer.isOpened():
            raise ValueError(f"Cannot create video writer: {self.output_path}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: BGR image array to write
            
        Raises:
            RuntimeError: If writer not opened via context manager
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter must be used as context manager")
        
        self._writer.write(frame)
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count
