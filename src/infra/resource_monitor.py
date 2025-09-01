"""
resource_monitor.py

System resource monitoring module for tracking CPU, memory, disk, and GPU usage.

This module provides a `ResourceMonitor` class that continuously tracks system resource usage
and logs warnings when predefined thresholds are exceeded. The monitor supports both CPU and
GPU (if available) usage statistics using `psutil` and optionally `pynvml`.

Core functionality:
- Monitor CPU, memory, disk, and GPU utilization in real time.
- Load usage thresholds and interval settings from a central Settings configuration.
- Log usage data and alerts with full-color console output and rotating file logs.
- Modular structure enables embedding into larger observability or watchdog systems.

Main components:
- `ResourceMonitor`: Central class for monitoring system resource usage.
- `start_monitoring()`: Main loop for continuous monitoring at configurable intervals.
- `check_cpu_usage()`, `check_memory_usage()`, `check_disk_usage()`, `check_gpu_usage()`: 
  Individual resource check functions with threshold alerting and error handling.

Key parameters from settings:
- `CPU_THRESHOLD`: Float (0.0–1.0) indicating CPU usage alert threshold.
- `MEMORY_THRESHOLD`: Float for memory usage alert threshold.
- `DISK_THRESHOLD`: Float for disk usage alert threshold.
- `GPUs_THRESHOLD`: Float for GPU usage alert threshold.
- `GPU_AVAILABLE`: Boolean toggle for enabling GPU monitoring.
- `MONITOR_INTERVAL`: Integer for time interval between resource checks (in seconds).

Dependencies:
- `psutil`: Used for retrieving CPU, memory, and disk statistics.
- `pynvml`: Optional. Used for querying NVIDIA GPU utilization.
- `src.helpers.Settings`: Application configuration loader.
- `src.infra.setup_logging`: Colorful and rotating logger setup utility.

Usage example:
    monitor = ResourceMonitor()
    monitor.start_monitoring()

Logging example:
    2025-07-21 18:25:43,102 - RESOURCE-MONITOR-CORE - INFO - CPU Usage: 15.37%
    2025-07-21 18:25:43,104 - RESOURCE-MONITOR-CORE - WARNING - High memory usage detected: 92.15%
    2025-07-21 18:25:43,106 - RESOURCE-MONITOR-CORE - INFO - Disk Usage: 55.21%

Notes:
- GPU usage tracking requires `pynvml` and a supported NVIDIA GPU.
- To gracefully stop monitoring, press Ctrl+C.
- Designed to be used standalone or integrated into larger observability dashboards.
"""

import time
import logging
import os
import sys
from typing import Dict, Union

import psutil

# Setup main directory for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ModuleNotFoundError, ImportError) as e:
    logging.error("Error setting up main directory: %s", e, exc_info=True)
    sys.exit(1)

from src.helpers import Settings, get_settings
from src.infra import setup_logging

logger = setup_logging(name="RESOURCE-MONITOR-CORE")


class ResourceMonitor:
    """
    Monitors system resources and logs alerts if thresholds are exceeded.

    Attributes:
        cpu_threshold (float): CPU usage threshold (0.0-1.0).
        memory_threshold (float): Memory usage threshold (0.0-1.0).
        disk_threshold (float): Disk usage threshold (0.0-1.0).
        gpu_threshold (float): GPU usage threshold (0.0-1.0).
        gpu_available (bool): Flag indicating whether GPU monitoring is enabled.
        app_settings (Settings): Loaded application settings instance.
    """

    def __init__(self) -> None:
        """
        Initialize the ResourceMonitor with thresholds from settings.
        """
        self.app_settings: Settings = get_settings()

        self.cpu_threshold: float = self.app_settings.MONITOR_CPU_THRESHOLD
        self.memory_threshold: float = self.app_settings.MONITOR_MEMORY_THRESHOLD
        self.disk_threshold: float = self.app_settings.MONITOR_DISK_THRESHOLD
        self.gpu_threshold: float = self.app_settings.MONITOR_GPU_THRESHOLD
        self.gpu_available: bool = getattr(self.app_settings, "GPU_AVAILABLE", False)

    def check_cpu_usage(self) -> Dict[str, Union[float, str]]:
        """
        Check current CPU usage.

        Returns:
            dict: Contains 'cpu_usage' as float percentage or 'error' string.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            logger.info(f"CPU Usage: {cpu_usage * 100:.2f}%")

            if cpu_usage > self.cpu_threshold:
                logger.warning(f"High CPU usage detected: {cpu_usage * 100:.2f}%")

            return {"cpu_usage": cpu_usage}
        except psutil.Error as err:
            logger.error(f"Failed to get CPU usage: {err}")
            return {"error": str(err)}

    def check_memory_usage(self) -> Dict[str, Union[float, str]]:
        """
        Check current memory usage.

        Returns:
            dict: Contains 'memory_usage' as float percentage or 'error' string.
        """
        try:
            memory = psutil.virtual_memory()
            mem_usage = memory.percent / 100.0
            logger.info(f"Memory Usage: {mem_usage * 100:.2f}%")

            if mem_usage > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory.percent:.2f}%")

            return {"memory_usage": mem_usage}
        except psutil.Error as err:
            logger.error(f"Failed to get memory usage: {err}")
            return {"error": str(err)}

    def check_disk_usage(self) -> Dict[str, Union[float, str]]:
        """
        Check current disk usage on root partition.

        Returns:
            dict: Contains 'disk_usage' as float percentage or 'error' string.
        """
        try:
            disk = psutil.disk_usage("/")
            disk_usage = disk.percent / 100.0
            logger.info(f"Disk Usage: {disk_usage * 100:.2f}%")

            if disk_usage > self.disk_threshold:
                logger.warning(f"High disk usage detected: {disk.percent:.2f}%")

            return {"disk_usage": disk_usage}
        except psutil.Error as err:
            logger.error(f"Failed to get disk usage: {err}")
            return {"error": str(err)}

    def check_gpu_usage(self) -> Dict[str, Union[float, str]]:
        """
        Check current GPU usage if available.

        Returns:
            dict: Contains 'gpu_usage' as float percentage, or error string.
        """
        if not self.gpu_available:
            logger.info("GPU monitoring is disabled or unavailable.")
            return {"gpu_usage": "GPU monitoring unavailable"}

        try:
            from pynvml import (
                nvmlInit,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetUtilizationRates,
            )

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = util.gpu / 100.0

            logger.info(f"GPU Usage: {gpu_usage * 100:.2f}%")

            if gpu_usage > self.gpu_threshold:
                logger.warning(f"High GPU usage detected: {gpu_usage * 100:.2f}%")

            return {"gpu_usage": gpu_usage}

        except ImportError:
            logger.error("pynvml library is not installed for GPU monitoring.")
            return {"error": "pynvml library not installed"}

        except Exception as err:
            logger.error(f"Failed to get GPU usage: {err}")
            return {"error": str(err)}

    def start_monitoring(self) -> None:
        """
        Continuously monitor system resources at configured intervals.
        """
        interval = self.app_settings.MONITOR_HEALTH_CHECK_INTERVAL_SEC
        logger.info("Starting system resource monitoring (interval: %d sec)...", interval)

        try:
            while True:
                self.check_cpu_usage()
                self.check_memory_usage()
                self.check_disk_usage()

                if self.gpu_available:
                    self.check_gpu_usage()

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Resource monitoring stopped by user.")
        except Exception as exc:
            logger.error(f"Unexpected error during monitoring: {exc}", exc_info=True)


if __name__ == "__main__":
    monitor = ResourceMonitor()

    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
