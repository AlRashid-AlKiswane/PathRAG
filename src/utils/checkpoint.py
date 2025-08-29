#!/usr/bin/env python3
"""
Checkpoint Callback Utility for PathRAG

Provides a reusable progress callback function that can be passed
into graph-building or long-running processes. This allows logging
and monitoring of progress without tightly coupling to the main loop.

Features:
- Logs percentage progress
- Handles invalid input safely
- Uses centralized logging via setup_logging
"""

from typing import Optional
from src.infra import setup_logging

# Configure logger
logger = setup_logging(name="CHECKPOINT-CALLBACK")


def checkpoint_callback(processed: int, total: int) -> Optional[float]:
    """
    Progress callback for graph building or batch processing.

    Args:
        processed (int): Number of items already processed.
        total (int): Total number of items to process.

    Returns:
        Optional[float]: Progress as a percentage (0.0–100.0) if valid,
        otherwise None.

    Notes:
        - Logs progress at the INFO level.
        - Guards against division by zero and invalid inputs.
        - Can be passed directly as a callback to PathRAG or other
          graph-building workflows.
    """
    try:
        if total <= 0:
            logger.warning("Invalid total items: %s", total)
            return None
        if processed < 0:
            logger.warning("Invalid processed items: %s", processed)
            return None

        progress = (processed / total) * 100
        logger.info("Graph building progress: %.1f%% (%s/%s)", progress, processed, total)
        return progress

    except Exception as e:
        logger.error("Error in checkpoint_callback: %s", str(e))
        return None
