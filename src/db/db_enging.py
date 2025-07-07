"""
SQLite Database Connection Module

This module provides robust functionality for creating and managing SQLite database connections
with comprehensive error handling and automatic directory creation.
"""

import logging
import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position, no-member, redefined-outer-name, logging-format-interpolation

from src.infra import setup_logging
from src.helpers import get_settings, Settings

# Initialize application settings and logger
logger = setup_logging()
app_settings: Settings = get_settings()

def get_sqlite_engine() -> Optional[sqlite3.Connection]:
    """
    Creates and returns a connection to an SQLite database with robust error handling.

    Returns:
        sqlite3.Connection: Database connection if successful
        None: If connection fails

    Raises:
        ValueError: For configuration issues
        sqlite3.Error: For database-specific errors
    """
    try:
        if not app_settings.SQLITE_DB:
            raise ValueError("SQLite database path is missing from configuration.")

        db_path = Path(app_settings.SQLITE_DB)
        db_dir = db_path.parent

        # Ensure directory exists with proper permissions
        try:
            db_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            logger.debug("Successfully created or verified directory: %s", db_dir)
        except PermissionError as pe:
            logger.error("Failed to create directory due to permission error: %s", pe)
            raise sqlite3.OperationalError(f"Directory creation failed: {pe}") from pe

        # Verify directory is writable
        if not os.access(db_dir, os.W_OK):
            logger.error("Directory is not writable: %s", db_dir)
            raise sqlite3.OperationalError(f"Directory not writable: {db_dir}")

        # Create database connection
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging
            logger.info("Successfully connected to SQLite database at: %s", db_path)
            return conn
        except sqlite3.Error as se:
            logger.error("SQLite connection failed: %s", se)
            raise

    except ValueError as ve:
        logger.error("Configuration error: %s", ve)
        raise
    except sqlite3.Error as se:
        logger.error("SQLite operational error: %s", se)
        raise
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpected error during SQLite engine creation: %s", e)
        return None

if __name__ == "__main__":
    conn = get_sqlite_engine()

    if conn:
        print("Connection Database Successfully")
        conn.close()
    else:
        print("Connection Filed")
