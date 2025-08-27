"""
pathrag_metrics.py
==================

This module provides a thread-safe metrics collection utility for PathRAG 
(Path-aware Retrieval-Augmented Generation) operations. It allows tracking 
and reporting of performance metrics such as graph build time, retrieval 
time, path pruning statistics, and caching efficiency.

Features
--------
- Thread-safe increments and updates to metrics.
- Tracks key performance indicators (KPIs) for PathRAG pipelines.
- Provides snapshot reports of collected metrics for logging or monitoring.
- Can be easily extended with new metric fields.
- Robust error handling with detailed logging.

Dependencies
------------
- threading : For thread-safe access to metrics.
- logging : For structured logging of errors and warnings.
- typing.Dict : For type annotations.

Example
-------
>>> from pathrag_metrics import PathRAGMetrics
>>> metrics = PathRAGMetrics()
>>> metrics.increment("nodes_processed", 10)
>>> metrics.set_metric("graph_build_time", 1.23)
>>> report = metrics.get_report()
>>> print(report)
{'graph_build_time': 1.23, 'retrieval_time': 0.0, 'path_pruning_time': 0.0,
 'nodes_processed': 10, 'edges_created': 0, 'paths_pruned': 0,
 'cache_hits': 0, 'cache_misses': 0}
"""

import logging
import os
import sys
import threading
from typing import Dict

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

from src.infra import setup_logging

logger = setup_logging(name="RAG-METRICS")


class PathRAGMetrics:
    """
    Metrics collection and reporting for PathRAG operations.

    This class provides a thread-safe way to collect and update performance 
    metrics across different stages of a PathRAG pipeline (e.g., graph 
    building, retrieval, pruning, caching). It ensures that metrics remain 
    consistent even when accessed concurrently by multiple threads.
    """

    def __init__(self):
        """Initialize metrics with default values and setup thread lock."""
        try:
            self.metrics = {
                "graph_build_time": 0.0,
                "retrieval_time": 0.0,
                "path_pruning_time": 0.0,
                "nodes_processed": 0,
                "edges_created": 0,
                "paths_pruned": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }
            self._lock = threading.Lock()
            logger.info("PathRAGMetrics initialized with default values.")
        except Exception as e:
            logger.error("Failed to initialize PathRAGMetrics: %s", e, exc_info=True)
            raise

    def increment(self, metric: str, value: float = 1.0):
        """
        Increment a metric by a given value in a thread-safe way.

        Parameters
        ----------
        metric : str
            The name of the metric to increment.
        value : float, optional
            The amount to increment the metric by (default is 1.0).
        """
        try:
            with self._lock:
                if metric not in self.metrics:
                    logger.error("Attempted to increment unknown metric: %s", metric)
                    raise KeyError(f"Unknown metric: {metric}")
                old_val = self.metrics[metric]
                self.metrics[metric] += value
                logger.debug(
                    "Incremented metric '%s' from %s to %s",
                    metric, old_val, self.metrics[metric]
                )
        except Exception as e:
            logger.error("Error incrementing metric '%s': %s", metric, e, exc_info=True)
            raise

    def set_metric(self, metric: str, value: float):
        """
        Set a metric to an explicit value in a thread-safe way.

        Parameters
        ----------
        metric : str
            The name of the metric to set.
        value : float
            The new value for the metric.
        """
        try:
            with self._lock:
                if metric not in self.metrics:
                    logger.error("Attempted to set unknown metric: %s", metric)
                    raise KeyError(f"Unknown metric: {metric}")
                old_val = self.metrics[metric]
                self.metrics[metric] = value
                logger.debug(
                    "Set metric '%s' from %s to %s", metric, old_val, value
                )
        except Exception as e:
            logger.error("Error setting metric '%s': %s", metric, e, exc_info=True)
            raise

    def get_report(self) -> Dict[str, float]:
        """
        Retrieve a snapshot of all current metrics.

        Returns
        -------
        Dict[str, float]
            A copy of the metrics dictionary with current values.
        """
        try:
            with self._lock:
                snapshot = self.metrics.copy()
            logger.info("Generated metrics report: %s", snapshot)
            return snapshot
        except Exception as e:
            logger.error("Failed to generate metrics report: %s", e, exc_info=True)
            return {}
