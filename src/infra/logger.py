"""
Logging module for application-wide logging with colored output.

This module provides a centralized logging system with four log levels:
- INFO (blue)
- DEBUG (green)
- WARNING (yellow)
- ERROR (red)

Logs are saved to 'app.log' in the logs directory.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    print(f"Failed to set up main directory path: {e}")
    sys.exit(1)

# Define log colors
COLORS = {
    "INFO": "\033[94m",    # Blue
    "DEBUG": "\033[92m",   # Green
    "WARNING": "\033[93m", # Yellow
    "ERROR": "\033[91m",   # Red
    "END": "\033[0m",      # Reset color
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""
    def format(self, record):
        message = super().format(record)
        return f"{COLORS.get(record.levelname, '')}{message}{COLORS['END']}"

# Global flag to avoid reinitializing
# pylint: disable=invalid-name
_logger_initialized = False

def setup_logging(
    name: str = "app_logger",
    log_dir=f"{MAIN_DIR}/logs",
    log_file="app.log",
    console_level=logging.DEBUG,
):
    """
    Set up logging configuration with colored console output and file logging.

    Args:
        log_dir (str): Directory to store log files.
        log_file (str): Name of the log file.
        console_level (int): Console log level (e.g., logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _logger_initialized

    logger__ = logging.getLogger(name=name)
    logger__.setLevel(logging.DEBUG)
    logger__.propagate = False  # Prevent propagation to root logger to avoid duplicates

    if _logger_initialized:
        return logger__

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Console handler (colored output)
    console_handler_exists = any(isinstance(h, logging.StreamHandler) for h in logger__.handlers)
    if not console_handler_exists:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger__.addHandler(console_handler)

    # File handler (rotating)
    file_handler_exists = any(isinstance(h, RotatingFileHandler) for h in logger__.handlers)
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger__.addHandler(file_handler)

    _logger_initialized = True
    return logger__


# Initialize the logger once at module import
logger = setup_logging()

# Example usage
def log_examples():
    """
    Example
    """
    logger.debug("This is a debug message - detailed technical information.")
    logger.info("This is an info message - general application flow.")
    logger.warning("This is a warning message - something unexpected happened.")
    logger.error("This is an error message - something failed.")

if __name__ == "__main__":
    log_examples()