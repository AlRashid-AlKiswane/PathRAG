"""
Chunks to Graph RAG API

This module defines a FastAPI route that retrieves chunked data from a SQLite table
and sends it to a LightRAG instance for entity and relation extraction.

Key Features:
- Uses dependency injection to get the database connection and LightRAG instance.
- Pulls chunk data from a specific table and columns.
- Extracts entities and relationships for graph-based RAG systems.
- Returns structured data as JSON.

Raises:
- HTTP 500 if data retrieval or entity extraction fails.
"""

import os
import sys
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlite3 import Connection

# Project path setup
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

# Imports after path setup
from src.rag import LightRAG
from src.db import pull_from_table
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_db_conn, get_light_rag

# Initialize logger and settings
logger = setup_logging()
app_settings: Settings = get_settings()

chunks_to_rag_route = APIRouter(
    prefix="/api/v1/chunks_rag",
    tags=["Chunks Store in Graph LightRAG"],
    responses={404: {"description": "Not found"}}
)


@chunks_to_rag_route.post("", response_class=JSONResponse)
async def chunks_to_rag(
    columns: List[str] = ["id", "chunk", "dataName"],
    table_name: str = "chunks",
    conn: Connection = Depends(get_db_conn),
    light_rag: LightRAG = Depends(get_light_rag),
):
    """
    Extracts entities and relations from chunked data and processes them via LightRAG.

    Args:
        columns (List[str]): List of columns to retrieve from the chunks table.
        table_name (str): SQLite table name containing the chunks.
        conn (Connection): SQLite connection from FastAPI dependency injection.
        light_rag (LightRAG): LightRAG instance for entity and relation extraction.

    Returns:
        JSONResponse: JSON containing extracted entities and relations.

    Raises:
        HTTPException: If chunk retrieval or RAG processing fails.
    """
    try:
        chunks = pull_from_table(conn=conn, columns=columns, table_name=table_name)
        if not chunks:
            raise HTTPException(status_code=500, detail="No chunk data retrieved from database.")

        entities, relations = light_rag.extract_entities_and_relations(chunks=chunks)

        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations.")

        return JSONResponse(
            content={
                "message": "Entity and relation extraction successful."
            },
            status_code=200
        )

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        logger.exception("Error during chunks-to-RAG processing.")
        raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")
