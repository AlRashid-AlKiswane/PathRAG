"""
infra package

Provides infrastructure-level utilities and services for the application, including:
- Logging setup with colorized console output and rotating file handlers.
- System resource monitoring (CPU, memory, disk, GPU) with configurable thresholds.

Modules:
- logger: Centralized logger configuration via `setup_logging()`.
- resource_monitor: Continuous system usage monitoring via `ResourceMonitor`.

Usage example:
    from src.infra import setup_logging, ResourceMonitor

    logger = setup_logging(name="APP")
    monitor = ResourceMonitor()
    monitor.start_monitoring()
"""

from .logger import setup_logging
from .resource_monitor import ResourceMonitor
from .memory_monitor import MemoryMonitor
__all__ = ["setup_logging", "ResourceMonitor"]
