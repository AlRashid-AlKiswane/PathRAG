"""
Database Table Mangemnt Module

Provides functions for safely clearing SQLite database tables with:
- Input validation
- Transaction managemnt
- Comprehensive error handling 
- Dettalied logging
"""

import os
import sys
import logging
import sqlite3

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.utils import setup_logging
from src.helpers import get_settings, Settings

# Initialize application settings and logger
logger = setup_logging()
app_settings: Settings = get_settings()


def clear_table(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Clears all records from the specified SQLite table.

    Args:
        conn: An active database connection
        table_name: The name of the table to clear

    Raises:
        ValueError: If the table name contains invalid characters
        RuntimeError: If an error occurs during deletion
        sqlite3.Error: For database-specific errors
    """
    if not isinstance(table_name, str) or not table_name.isidentifier():
        error_msg = f"Invalid table name: {table_name}"
        logger.error("Invalid table name: %s", table_name)
        raise ValueError(error_msg)

    cursor = None
    try:
        logger.debug("Starting to clear table '%s'", table_name)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
        logger.info("Successfully cleared table '%s'", table_name)

    except sqlite3.OperationalError as e:
        error_msg = f"Operational error clearing table '{table_name}': {e}"
        logger.error("Operational error clearing table '%s': %s", table_name, e)
        raise RuntimeError(error_msg) from e

    except sqlite3.DatabaseError as e:
        error_msg = f"Database integrity error clearing table '{table_name}': {e}"
        logger.error("Database integrity error clearing table '%s': %s", table_name, e)
        raise RuntimeError(error_msg) from e

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Unexpected error clearing table '{table_name}': {e}"
        logger.exception("Unexpected error clearing table '%s': %s", table_name, e)
        raise RuntimeError(error_msg) from e

    finally:
        if cursor:
            try:
                cursor.close()
                logger.debug("Cursor closed for table '%s'", table_name)
            except sqlite3.Error as e:
                logger.warning("Error closing cursor: %s", e)
