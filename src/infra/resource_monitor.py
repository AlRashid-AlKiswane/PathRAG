"""
System Resource Monitoring Module.

This module provides the ResourceMonitor class which monitors CPU, memory,
disk, and GPU usage. It uses psutil and pynvml libraries to gather system
metrics and logs alerts when usage exceeds configured thresholds.

Usage:
    monitor = ResourceMonitor()
    monitor.start_monitoring()

Note:
    GPU monitoring requires NVIDIA drivers and pynvml package installed.
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

logger = setup_logging()


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

        self.cpu_threshold: float = self.app_settings.CPU_THRESHOLD
        self.memory_threshold: float = self.app_settings.MEMORY_THRESHOLD
        self.disk_threshold: float = self.app_settings.DISK_THRESHOLD
        self.gpu_threshold: float = self.app_settings.GPUs_THRESHOLD
        self.gpu_available: bool = getattr(self.app_settings, "GPU_AVAILABLE", False)

    def check_cpu_usage(self) -> Dict[str, Union[float, str]]:
        """
        Check current CPU usage.

        Returns:
            dict: Contains 'cpu_usage' as float percentage or 'error' string.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
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
        interval = self.app_settings.MONITOR_INTERVAL
        logger.info("Starting system resource monitoring with interval %d seconds.", interval)

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
