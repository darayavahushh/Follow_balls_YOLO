"""
=============================================================================
Logging Utilities
=============================================================================
Configures logging to file and console with different verbosity levels.

Console: Reduced verbosity (INFO level, minimal formatting)
File: Full details (DEBUG level, timestamps, comprehensive formatting)

Usage:
    from tools.logging_utils import setup_logger, get_logger
    
    logger = setup_logger("train_2d", log_path, config)
    logger.info("Starting training...")
=============================================================================
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import io


# =============================================================================
# CUSTOM FORMATTERS
# =============================================================================

class ConsoleFormatter(logging.Formatter):
    """
    Minimal formatter for console output.
    
    Shows only the message for INFO level, includes level for others.
    Uses colors for different levels (if terminal supports it).
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[0m',       # Default (no color)
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and self._supports_color()
    
    @staticmethod
    def _supports_color() -> bool:
        """Check if terminal supports colors."""
        if not hasattr(sys.stdout, 'isatty'):
            return False
        if not sys.stdout.isatty():
            return False
        if os.environ.get('NO_COLOR'):
            return False
        return True
    
    def format(self, record: logging.LogRecord) -> str:
        # For INFO, just show the message
        if record.levelno == logging.INFO:
            return record.getMessage()
        
        # For other levels, show level prefix
        level = record.levelname
        message = record.getMessage()
        
        if self.use_colors:
            color = self.COLORS.get(level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            return f"{color}[{level}]{reset} {message}"
        else:
            return f"[{level}] {message}"


class FileFormatter(logging.Formatter):
    """
    Detailed formatter for log files.
    
    Includes timestamp, level, and full message.
    """
    
    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_logger(
    name: str,
    log_path: Path,
    config: Optional[Dict[str, Any]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name (e.g., "train_2d")
        log_path: Path to log file
        config: Optional config to log at start
        console_level: Logging level for console (default: INFO)
        file_level: Logging level for file (default: DEBUG)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all, handlers filter
    
    # Remove existing handlers (in case of re-initialization)
    logger.handlers.clear()
    
    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler - detailed output
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)
    
    # Console handler - minimal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)
    
    # Log header
    _log_header(logger, name, config)
    
    return logger


def _log_header(
    logger: logging.Logger,
    operation_name: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log header with operation info and config summary.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation
        config: Configuration dictionary
    """
    separator = "=" * 80
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log to file (detailed)
    logger.debug(separator)
    logger.debug(f"BALL CV CHALLENGE - {operation_name.upper()}")
    logger.debug(separator)
    logger.debug(f"Timestamp: {timestamp}")
    logger.debug(separator)
    
    if config:
        logger.debug("")
        logger.debug("CONFIGURATION SUMMARY")
        logger.debug("-" * 40)
        _log_config_summary(logger, config)
        logger.debug("")
        logger.debug(separator)
    
    logger.debug("")
    logger.debug("EXECUTION LOG")
    logger.debug("-" * 40)


def _log_config_summary(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log a summary of the configuration.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    # Model config
    if 'model' in config:
        model = config['model']
        logger.debug(f"Model: {model.get('name', 'N/A')}")
        logger.debug(f"Image Size: {model.get('image_size', 'N/A')}")
        logger.debug(f"Confidence Threshold: {model.get('confidence_threshold', 'N/A')}")
    
    # Training config
    if 'training' in config:
        training = config['training']
        logger.debug(f"Epochs: {training.get('epochs', 'N/A')}")
        logger.debug(f"Batch Size: {training.get('batch_size', 'N/A')}")
        logger.debug(f"Patience: {training.get('patience', 'N/A')}")
    
    # Classes config
    if 'classes' in config:
        classes = config['classes']
        logger.debug(f"Classes: {classes.get('names', 'N/A')}")
    
    # Paths config
    if 'paths' in config:
        paths = config['paths']
        logger.debug(f"Input Video: {paths.get('input_video', 'N/A')}")
        logger.debug(f"Dataset: {paths.get('source_dataset', 'N/A')}")


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# SUMMARY LOGGING
# =============================================================================

def log_summary(
    logger: logging.Logger,
    status: str,
    duration_sec: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log operation summary at the end.
    
    Args:
        logger: Logger instance
        status: Status string (e.g., "SUCCESS", "FAILED")
        duration_sec: Operation duration in seconds
        metrics: Optional metrics dictionary
    """
    separator = "=" * 80
    
    logger.debug("")
    logger.debug(separator)
    logger.debug("SUMMARY")
    logger.debug("-" * 40)
    logger.debug(f"Status: {status}")
    
    if duration_sec is not None:
        hours, remainder = divmod(int(duration_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.debug(f"Duration: {hours}h {minutes}m {seconds}s")
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.debug(f"{key}: {value:.4f}")
            else:
                logger.debug(f"{key}: {value}")
    
    logger.debug(separator)


# =============================================================================
# PROGRESS LOGGING
# =============================================================================

class ProgressLogger:
    """
    Helper for logging progress updates.
    
    Logs every N steps to file, but only periodic updates to console.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        prefix: str = "Progress",
        console_interval: int = 50,
        file_interval: int = 10
    ):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total: Total number of steps
            prefix: Prefix for progress messages
            console_interval: Log to console every N steps
            file_interval: Log to file every N steps
        """
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.console_interval = console_interval
        self.file_interval = file_interval
        self.current = 0
    
    def update(self, current: Optional[int] = None, extra: str = "") -> None:
        """
        Update progress.
        
        Args:
            current: Current step (if None, increments by 1)
            extra: Extra information to append
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percentage = (self.current / self.total) * 100
        message = f"{self.prefix}: {self.current}/{self.total} ({percentage:.1f}%)"
        
        if extra:
            message += f" - {extra}"
        
        # Log to file more frequently
        if self.current % self.file_interval == 0:
            self.logger.debug(message)
        
        # Log to console less frequently
        if self.current % self.console_interval == 0:
            self.logger.info(f"   {message}")


# =============================================================================
# STDOUT CAPTURE
# =============================================================================

class LoggerWriter:
    """
    Writer that redirects output to a logger.
    
    Useful for capturing print statements from external libraries.
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        self.logger = logger
        self.level = level
        self.buffer = ""
    
    def write(self, message: str) -> None:
        if message and message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self) -> None:
        pass


@contextmanager
def capture_output(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Context manager to capture stdout/stderr to logger.
    
    Args:
        logger: Logger instance
        level: Log level for captured output
        
    Usage:
        with capture_output(logger):
            external_library.train()  # Output goes to logger
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        sys.stdout = LoggerWriter(logger, level)
        sys.stderr = LoggerWriter(logger, logging.WARNING)
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# =============================================================================
# DECORATORS
# =============================================================================

def log_function_call(logger: logging.Logger):
    """
    Decorator to log function entry and exit.
    
    Args:
        logger: Logger instance
        
    Usage:
        @log_function_call(logger)
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
