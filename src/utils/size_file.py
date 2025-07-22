"""
File Utility Module

This module provides a utility function for calculating the size of a file in gigabytes (GB).
It includes error handling and structured logging.

Author: Alrashid AlKiswane
Date: 2025-07-07
"""

import os
import sys
import logging
from typing import Tuple
from pathlib import Path


# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging

logger = setup_logging(name="SIZE-FILE")


def get_size(path: str) -> Tuple[str, float]:
    """
    Calculates the size of a file in gigabytes (GB).

    Args:
        path (str): The file path to calculate the size of.

    Returns:
       Tuple[str, float].

    Raises:
        FileNotFoundError: If the given path does not exist or is not a file.
        OSError: If there's an issue accessing the file size.
        Exception: Catches any unexpected exception during size retrieval.

    Example:
        >>> get_size("/path/to/some/file.txt")
        'File size: 0.01 GB'
    """
    try:
        path_obj = Path(path)

        if not path_obj.exists() or not path_obj.is_file():
            msg = f"The path does not exist or is not a valid file: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        file_size_bytes = path_obj.stat().st_size
        file_size_gb = file_size_bytes / (1024 ** 3)

        result = f"File size: {file_size_gb:.2f} GB"
        logger.info(result)
        return result, file_size_gb

    except FileNotFoundError as fnf_error:
        logger.exception("FileNotFoundError: %s", fnf_error)
        raise

    except OSError as os_error:
        logger.exception("OSError while accessing file size: %s", os_error)
        raise

    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
        raise

if __name__ == "__main__":
    path = ...
    msg, size = get_size(path=path)
    print(msg, size)
