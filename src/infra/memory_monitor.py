"""
memory_monitor.py
=================

This module provides a utility class for monitoring and enforcing memory usage 
limits within a Python application. It is designed for use in long-running or 
resource-intensive systems such as Retrieval-Augmented Generation (RAG) pipelines, 
data processing workflows, or machine learning model serving environments.

Features
--------
- Monitor the memory usage of the current process in gigabytes.
- Define a configurable memory usage limit (in GB).
- Check whether the process is within the memory usage limit.
- Use a context manager (`memory_guard`) to automatically log warnings when 
  operations exceed memory constraints.
- Log informative messages about memory growth during monitored operations.

Dependencies
------------
- psutil : For querying process memory information.
- logging : For logging warnings and info messages.

Example
-------
>>> from memory_monitor import MemoryMonitor
>>> monitor = MemoryMonitor(limit_gb=2)
>>> with monitor.memory_guard():
...     big_data = [0] * (10**7)   # Example memory-heavy operation
# If memory usage exceeds 2 GB, a warning will be logged.
"""

from contextlib import contextmanager
import logging
import psutil

class MemoryMonitor:
    """
    Monitor and enforce memory usage limits for processes.

    Attributes
    ----------
    limit_bytes : int
        Memory usage limit in bytes.
    process : psutil.Process
        Current process being monitored.
    """

    def __init__(self, limit_gb: float = 4.0):
        """
        Initialize the memory monitor.

        Parameters
        ----------
        limit_gb : float, optional
            Maximum memory usage allowed in gigabytes (default is 4.0).
        """
        self.limit_bytes = int(limit_gb * 1024**3)
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """
        Get current process memory usage in gigabytes.

        Returns
        -------
        float
            Current memory usage in GB.
        """
        return self.process.memory_info().rss / 1024**3

    def check_memory_limit(self) -> bool:
        """
        Check if current memory usage is within the configured limit.

        Returns
        -------
        bool
            True if usage is below the limit, False otherwise.
        """
        return self.process.memory_info().rss < self.limit_bytes

    @contextmanager
    def memory_guard(self):
        """
        Context manager to monitor memory usage during an operation.

        Logs a warning if the memory usage exceeds the configured limit.

        Usage
        -----
        >>> monitor = MemoryMonitor(limit_gb=2)
        >>> with monitor.memory_guard():
        ...     big_data = [0] * (10**7)
        """
        initial = self.get_memory_usage()
        try:
            yield
        finally:
            current = self.get_memory_usage()
            if self.process.memory_info().rss > self.limit_bytes:
                logging.warning(
                    f"Memory usage exceeded limit: {current:.2f} GB "
                    f"(limit: {self.limit_bytes / 1024**3:.2f} GB)"
                )
            elif current > initial:
                logging.info(
                    f"Memory increased during operation: "
                    f"{initial:.2f} GB → {current:.2f} GB"
                )
