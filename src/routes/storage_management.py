"""
storage_management_route.py

This module defines an API route for managing storage tables in the system's SQLite database.
It provides options to clear specific tables or perform a full reset of all relevant data tables.

Endpoint:
    POST /api/v1/storage-management/

Parameters:
    - do_erase_all (bool): Clears all relevant tables: `chunks`, `embed_vector`, and `chatbot`.
    - reset_chunks (bool): Clears only the `chunks` table (default: True).
    - reset_embeddings (bool): Clears only the `embed_vector` table.
    - reset_chatbot_history (bool): Clears only the `chatbot` table.

Returns:
    JSONResponse: A message indicating which tables were successfully reset.
    - 200: Tables were cleared successfully.
    - 400: No reset flags were provided; no action taken.
    - 500: A database or unexpected server error occurred.

Functionality:
    - Uses dependency-injected SQLite connection.
    - Calls `clear_table()` utility from `src.db` to delete table contents.
    - Logs each operation at info/debug/warning levels.
    - Handles database and operational errors gracefully.

Dependencies:
    - FastAPI
    - SQLite
    - Custom project utilities (`clear_table`, `setup_logging`, `get_db_conn`)
"""

import os
import sys
import logging
from sqlite3 import Connection, OperationalError, DatabaseError

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.db import clear_table
from src.infra import setup_logging
from src import get_db_conn

# === Logger Setup ===
logger = setup_logging(name="MANAGEMENT-STORAGE")

# === API Router ===
storage_management_route = APIRouter(
    prefix="/api/v1/management-storage",
    tags=["Management Storage"],
    responses={404: {"description": "Not found"}}
)


@storage_management_route.post("")
async def storage_management(
    do_erase_all: bool = False,
    reset_chunks: bool = True,
    reset_embeddings: bool = False,
    reset_chatbot_history: bool = False,
    conn: Connection = Depends(get_db_conn)
):
    """
    Manage and optionally reset storage tables in the database.

    Args:
        do_erase_all (bool): If True, clears all relevant tables: 'chunks', 'embed_vector', 'chatbot'.
        reset_chunks (bool): If True, clears only the 'chunks' table.
        reset_embeddings (bool): If True, clears only the 'embed_vector' table.
        reset_chatbot_history (bool): If True, clears only the 'chatbot' table.
        conn (Connection): SQLite database connection, injected by dependency.

    Returns:
        JSONResponse: Operation result with a list of affected tables and success message.

    Raises:
        HTTPException: With appropriate status code for database or unexpected errors.
    """
    try:
        affected_tables = []

        if do_erase_all:
            tables = ["chunks", "embed_vector", "chatbot"]
            logger.info("Initiating full reset of all storage tables.")
            for table in tables:
                clear_table(conn=conn, table_name=table)
                affected_tables.append(table)
                logger.debug("Cleared table: %s", table)

        else:
            if reset_chunks:
                clear_table(conn=conn, table_name="chunks")
                affected_tables.append("chunks")
                logger.debug("Cleared table: chunks")

            if reset_embeddings:
                clear_table(conn=conn, table_name="embed_vector")
                affected_tables.append("embed_vector")
                logger.debug("Cleared table: embed_vector")

            if reset_chatbot_history:
                clear_table(conn=conn, table_name="chatbot")
                affected_tables.append("chatbot")
                logger.debug("Cleared table: chatbot")

        if not affected_tables:
            logger.warning("No table reset flags set. No tables were cleared.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "No reset flag provided. Specify at least one table to reset."}
            )

        logger.info("Storage management operation completed. Tables affected: %s", affected_tables)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Tables reset successfully.",
                "tables_affected": affected_tables
            }
        )

    except (OperationalError, DatabaseError) as db_err:
        logger.error("Database error during table reset: %s", db_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred during table reset."
        )

    except Exception as e:
        logger.critical("Unexpected error in storage management: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during storage management."
        )
