"""
Storage Management API Route

This module provides a FastAPI route to manage the underlying SQLite storage 
used for chunked documents, embeddings, and named entities.

Features:
- Clear one or more database tables
- Toggle reset on specific tables: chunks, embeddings, entities
- Log all operations with proper severity levels
- Graceful error handling with descriptive messages

Author: [Your Name]
Date: 2025-07-09
"""

import asyncio
import os
import sys
import logging
import json
from typing import List
from sqlite3 import Connection, OperationalError, DatabaseError

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("🔴 Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.db import clear_table
from src.infra import setup_logging
from src import get_db_conn

# === Logger & Settings ===
logger = setup_logging()

# === API Router ===
storage_management_route = APIRouter(
    prefix="/api/v1/storage-management",
    tags=["Storage management"],
    responses={404: {"description": "Not found"}}
)


@storage_management_route.post("")
async def storage_management(
    do_erase_all: bool = False,
    reset_chunks: bool = True,
    reset_embeddings: bool = False,
    reset_entities: bool = False,
    conn: Connection = Depends(get_db_conn)
):
    """
    Manage and optionally reset storage tables in the database.

    Args:
        do_erase_all (bool): If True, clears all relevant tables: chunks, vectors_embedding, and entities.
        reset_chunks (bool): If True, clears only the 'chunks' table.
        reset_embeddings (bool): If True, clears only the 'vectors_embedding' table.
        reset_entities (bool): If True, clears only the 'entities' table.
        conn (Connection): SQLite database connection, injected by dependency.

    Returns:
        JSONResponse: Operation summary with status message.
    """
    try:
        affected_tables = []

        if do_erase_all:
            tables = ["chunks", "vectors_embedding", "entities"]
            logger.warning("🔄 Full database reset initiated.")
            for table in tables:
                clear_table(conn=conn, table_name=table)
                affected_tables.append(table)
                logger.info("✅ Table '%s' cleared.", table)
            logger.debug("Tables cleared in erase-all mode: %s", affected_tables)

        else:
            if reset_chunks:
                clear_table(conn=conn, table_name="chunks")
                affected_tables.append("chunks")
                logger.info("✅ Table 'chunks' cleared.")

            if reset_embeddings:
                clear_table(conn=conn, table_name="vectors_embedding")
                affected_tables.append("vectors_embedding")
                logger.info("✅ Table 'vectors_embedding' cleared.")

            if reset_entities:
                clear_table(conn=conn, table_name="entities")
                affected_tables.append("entities")
                logger.info("✅ Table 'entities' cleared.")

        if not affected_tables:
            logger.warning("⚠️ No table reset flags were set to True. No action performed.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "No reset flag provided. Specify at least one table to reset."}
            )

        logger.debug("🧾 Final list of affected tables: %s", affected_tables)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Tables reset successfully.",
                "tables_affected": affected_tables
            }
        )

    except (OperationalError, DatabaseError) as db_err:
        logger.error("❌ Database operation failed: %s", db_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred during table reset."
        )
    except Exception as e:
        logger.critical("🔥 Unexpected error in storage management: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during storage management."
        )
