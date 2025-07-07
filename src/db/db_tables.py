"""
Database Initialization Module

This module provides functionality for initializing SQLite database tables
for saving chunks in a Retrieval-Augmented Generation (RAG) application.

It handles the creation of:
- chunks table: Stores document chunks with metadata
"""

import logging
import os
import sys
import sqlite3

# Set up the main directory for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.getLogger(__name__).error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings

logger = setup_logging()
app_settings: Settings = get_settings()


def init_chunks_table(conn: sqlite3.Connection) -> None:
    """
    Initialize the 'chunks' table for storing document chunks and metadata.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.

    Raises:
        sqlite3.Error: If table creation fails.
    """
    try:
        logger.info("Creating 'chunks' table if it doesn't exist...")

        create_query = """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT NOT NULL,
            file TEXT NOT NULL,
            dataName TEXT NOT NULL
        );
        """
        conn.execute(create_query)
        conn.commit()

        logger.info("'chunks' table created successfully.")

    except sqlite3.Error as e:
        logger.error("Failed to create 'chunks' table: %s", e)
        raise

if __name__ == "__main__":
    from src.db import get_sqlite_engine

    conn = get_sqlite_engine()
    if conn:
        logger.info("Initializing the database...")
        try:
            init_chunks_table(conn=conn)
            logger.info("Initialization completed.")
        finally:
            conn.close()
            logger.info("Database connection closed.")
    else:
        logger.error("Database connection failed.")
        sys.exit(1)
