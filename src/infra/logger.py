"""
logger.py

Application-wide logging utility with color-coded console output and rotating file logging.

Core functionality:
- Provides a centralized and reusable logger across modules with consistent formatting.
- Colorizes logs in the console based on severity (INFO, DEBUG, WARNING, ERROR).
- Stores logs persistently in a rotating file handler located under the 'logs/' directory.
- Prevents redundant re-initialization of loggers by caching by name.

Main components:
- `setup_logging()`: Initializes and returns a logger with colored console output and file logging.
- `ColoredFormatter`: Custom log formatter that adds ANSI colors to terminal messages.
- `log_examples()`: Demonstrates how to log messages at various levels using the configured logger.

Key features:
- Supports named loggers for modular traceability across files and components.
- Colored console messages using ANSI escape codes:
    - INFO (Blue)
    - DEBUG (Green)
    - WARNING (Yellow)
    - ERROR (Red)
- Log files stored in: `<project_root>/logs/app.log`
    - Max size: 50MB
    - Backup count: 5 rotating files
- Uses `RotatingFileHandler` for disk-efficient log file management.

Usage:
    from src.infra.logger import setup_logging

    logger = setup_logging(name="MY_MODULE")
    logger.info("Initialization complete.")
    logger.error("An unexpected error occurred.")

Example:
    $ python logger.py
    2025-07-21 18:12:00,123 - TEST - DEBUG - This is a debug message - detailed technical information.
    2025-07-21 18:12:00,124 - TEST - INFO - This is an info message - general application flow.
    2025-07-21 18:12:00,124 - TEST - WARNING - This is a warning message - something unexpected happened.
    2025-07-21 18:12:00,124 - TEST - ERROR - This is an error message - something failed.

Notes:
- This module is typically used at the infrastructure or utility level and should be imported
  once and reused throughout the application.
- Designed to work across environments including development, testing, and production.
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

# Prevent logger re-initialization
_logger_initialized = {}

def setup_logging(
    name: str = "app_logger",
    log_dir: str = f"{MAIN_DIR}/logs",
    log_file: str = "app.log",
    console_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Set up logging configuration with colored console output and file logging.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to store log files.
        log_file (str): Name of the log file.
        console_level (int): Console log level (e.g., logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if name in _logger_initialized:
        return _logger_initialized[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _logger_initialized[name] = logger
    return logger

# Initialize named logger
logger = setup_logging(name="TEST")

def log_examples():
    """Example usage of logger at various levels."""
    logger.debug("This is a debug message - detailed technical information.")
    logger.info("This is an info message - general application flow.")
    logger.warning("This is a warning message - something unexpected happened.")
    logger.error("This is an error message - something failed.")

if __name__ == "__main__":
    log_examples()
