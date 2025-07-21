"""
resource_monitor_route.py

This module defines a FastAPI route for monitoring and reporting system resource usage.

It leverages a custom `ResourceMonitor` utility to collect real-time statistics
about the system's CPU, memory, disk, and (if available) GPU usage.

Route:
    GET /api/v1/resource/

Returns:
    JSONResponse: A structured dictionary containing usage metrics for:
        - CPU (% utilization)
        - Memory (% usage, total, available)
        - Disk (% usage, total, used, free)
        - GPU (% utilization, memory stats) if supported

Features:
    - Logs resource usage request and status.
    - Warns if critical thresholds are exceeded (handled inside ResourceMonitor).
    - Gracefully handles internal monitoring errors with structured HTTP exceptions.

Raises:
    - HTTPException (500): If an unexpected error occurs during monitoring.

Dependencies:
    - ResourceMonitor (from `src.infra`)
    - FastAPI

Author:
    ALRashid AlKiswane
"""

import os
import sys
import logging

from fastapi import (APIRouter,
                     HTTPException,
                     status)

from fastapi.responses import JSONResponse

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging, ResourceMonitor
from src.helpers import get_settings, Settings

# Initialize logger and settings
logger = setup_logging(name="RESOURCE-MONITOR")
app_settings: Settings = get_settings()

resource_monitor_router = APIRouter(
    prefix="/api/v1/resource",
    tags=["Resources"],
    responses={404: {"description": "Not found"}},
)


@resource_monitor_router.get("/", response_class=JSONResponse, status_code=200)
async def get_resource_usage():
    """
    Get current system resource usage statistics.

    Returns:
        JSONResponse: Dictionary with current usage of CPU, memory, disk,
        and GPU (if available). Logs warning if usage exceeds thresholds.

    Raises:
        HTTPException: If monitoring fails due to internal error.
    """
    try:
        monitor = ResourceMonitor()
        logger.info("Received request to fetch system resource usage")

        usage_data = {
            "cpu": monitor.check_cpu_usage(),
            "memory": monitor.check_memory_usage(),
            "disk": monitor.check_disk_usage(),
        }

        if True:
            usage_data["gpu"] = monitor.check_gpu_usage()
        else:
            usage_data["gpu"] = {"gpu_usage": "GPU monitoring disabled"}

        logger.info("Successfully fetched resource usage")
        return JSONResponse(content=usage_data)

    except Exception as err:
        logger.error(f"Failed to retrieve resource usage: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching resource usage",
        )
