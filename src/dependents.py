"""
Database Connection Utility

This module provides a utility function for retrieving the active SQLite
database connection from the FastAPI application's state. It ensures robust
error handling and proper logging for both expected and unexpected issues.

Key Features:
- Sets up project directory for relative imports
- Retrieves the database connection from `request.app.state.conn`
- Handles missing or invalid connections gracefully using HTTP exceptions
- Logs debug and error messages during retrieval

Typical Usage:
    from fastapi import Request
    from src.db.utils import get_db_conn

    @app.get("/example")
    def example_endpoint(request: Request):
        conn = get_db_conn(request)
        # Perform DB operations...

Raises:
- HTTP 503 if the database connection is unavailable
- HTTP 500 for unexpected internal errors
"""



# Standard library imports
import logging
import os
import sys
from typing import Any
import sqlite3

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
    logging.debug("Main directory path configured: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

# Third-party imports
from fastapi import Request, HTTPException
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE
)

# Local application imports
from src.infra import setup_logging

# Initialize logger and application settings
logger = setup_logging()

def get_db_conn(request: Request) -> sqlite3.Connection:
    """
    Retrieve the SQLite database connection from the FastAPI app state.

    Args:
        request: The incoming FastAPI request object.

    Returns:
        sqlite3.Connection: Active SQLite connection.

    Raises:
        HTTPException: If the connection is missing or invalid (503 Service Unavailable)
    """
    try:
        conn = getattr(request.app.state, "conn", None)
        if not conn:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service unavailable."
            )
        logger.debug("Database connection retrieved successfully")
        return conn
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving database connection")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing database."
        ) from e
