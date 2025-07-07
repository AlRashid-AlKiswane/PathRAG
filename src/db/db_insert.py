"""
Chunk Insertion Module for RAG Database

This module provides functionality to insert individual document chunks into the
`chunks` table of an SQLite database. It is part of the document preprocessing
pipeline for a Retrieval-Augmented Generation (RAG) application.

Key Features:
- Inserts a document chunk along with its source file and data group name
- Handles database commit and rollback
- Uses lazy logging for efficient and clear runtime feedback
- Robust error handling to prevent data corruption

Functions:
- insert_chunk(): Safely inserts a single chunk into the database

Intended Usage:
- Used after document parsing and chunking to persist data
- Can be integrated with pipelines or called standalone for testing

Dependencies:
- SQLite3
- Custom settings and logging utilities from `src.helpers` and `src.infra`

Example:
    from src.db import insert_chunk
    conn = get_sqlite_engine()
    insert_chunk(conn, "chunk text", "file.pdf", "DocumentGroup")
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

logger = setup_logging()

def insert_chunk(conn: sqlite3.Connection,
                 chunk: str = None,
                 file: str = None,
                 dataName: str = None) -> bool:
    """
    Inserts a document chunk into the 'chunks' table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        chunk (str): The text content of the chunk.
        file (str): The file path or name from which the chunk was extracted.
        dataName (str): The source or logical group the chunk belongs to.

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (chunk, file, dataName)
            VALUES (:chunk, :file, :dataName)
            """,
            {"chunk": chunk, "file": file, "dataName": dataName}
        )
        conn.commit()
        logger.info("Inserted chunk into 'chunks' table from file: %s", file)
        return True

    except sqlite3.Error as e:
        logger.error("Failed to insert chunk: %s", e)
        conn.rollback()
        return False


if __name__ == "__main__":
    from src.db import get_sqlite_engine

    conn = get_sqlite_engine()
    if conn:
        success = insert_chunk(
            conn=conn,
            chunk="Sample chunk text",
            file="example.pdf",
            dataName="ImmigrationGuide2025"
        )
        logger.info("Insertion success: %s", success)
        conn.close()
