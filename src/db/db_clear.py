"""
Module for safely clearing tables in the SQLite database.

This module provides utility functions to remove all records from a specified table
within a SQLite database connection. It includes rigorous validation of table names,
error handling for SQLite exceptions, and detailed logging for operational transparency.

Usage:
    - Import `clear_table` and provide it with an active SQLite connection and a valid table name.
    - Ensure that the table name is a valid identifier to prevent SQL injection risks.
    - Logging is performed at various levels (DEBUG, INFO, ERROR) to trace execution and errors.

Example:
    >>> import sqlite3
    >>> from src.infra.db_clear import clear_table
    >>> conn = sqlite3.connect("mydatabase.db")
    >>> clear_table(conn, "users")
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
from src.infra import setup_logging
from src.helpers import get_settings, Settings

# Initialize application settings and logger
logger = setup_logging(name="DATABASE-CLEAR")
app_settings: Settings = get_settings()


def clear_table(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Delete all records from a specified table within the given SQLite database connection.

    This function validates the table name to ensure it is a proper identifier, avoiding SQL injection
    vulnerabilities. It then executes a DELETE operation on the entire table and commits the transaction.
    Extensive error handling is included to catch and log operational and database integrity issues,
    as well as any unexpected exceptions.

    Args:
        conn (sqlite3.Connection): An active SQLite database connection object.
        table_name (str): The name of the table to clear. Must be a valid Python identifier.

    Raises:
        ValueError: If the table name is invalid (e.g., contains special characters or is empty).
        RuntimeError: If a database operational or integrity error occurs during deletion.
        sqlite3.Error: For other SQLite-specific errors.

    Side Effects:
        Commits the transaction on successful deletion.
        Logs messages at DEBUG, INFO, WARNING, and ERROR levels.
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
