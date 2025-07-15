"""
NER Extraction API - FastAPI Module

This module defines a POST endpoint to extract Named Entities from text
using a lightweight HuggingFace-based NER model (via `NERModel` class).

Features:
- Fast inference using DistilBERT NER models.
- Multilingual support depending on model.
- Structured logging and full error handling.
- Compatible with LightRAG's graph-based entity storage.

Raises:
- HTTP 400 for invalid input.
- HTTP 500 for unexpected model errors.
"""

import os
import sys
import logging
from typing import List

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import sqlite3
# Project path setup
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("🔴 Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.db import pull_from_table, insert_ner_entities
from src.infra import setup_logging
from src.llms_providers import NERModel
from src import get_db_conn, get_ner_model

logger = setup_logging()



# === FastAPI Router ===
ner_route = APIRouter(
    prefix="/api/v1/ner",
    tags=["Chunks → NER Extraction"],
    responses={404: {"description": "Not found"}}
)

@ner_route.post("", response_class=JSONResponse)
async def extract_named_entities(
    columns: List[str] = ["id", "chunk", "dataName"],
    table_name: str = "chunks",
    conn: sqlite3.Connection = Depends(get_db_conn),
    ner_model: NERModel = Depends(get_ner_model)
) -> JSONResponse:
    """
    Extract named entities from database chunks using the NER model,
    and insert the results into the 'ner_entities' table.

    Args:
        columns (list): Columns to fetch from the chunk table.
        table_name (str): Table name where chunks are stored.
        conn (sqlite3.Connection): SQLite DB connection.
        ner_model (NERModel): Dependency-injected NER model.

    Returns:
        JSONResponse: Count of processed chunks and inserted entities.
    """
    logger.info("📦 Retrieving chunks from table '%s' with columns: %s", table_name, columns)

    try:
        # Pull chunks from database
        meta_chunks = pull_from_table(conn=conn, columns=columns, table_name=table_name)

        if not meta_chunks:
            logger.warning("⚠️ No records found in table '%s'.", table_name)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No chunks found in the specified database table."
            )

        valid_chunks = [row for row in meta_chunks if row.get("chunk")]
        if not valid_chunks:
            logger.warning("⚠️ No valid 'chunk' fields available in retrieved rows.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No usable chunk data found."
            )

        logger.debug("🧩 Retrieved %d valid chunks. First chunk preview: %s...",
                     len(valid_chunks), valid_chunks[0]['chunk'][:100])

        entity_total = 0

        for i, meta in enumerate(valid_chunks):
            chunk: str = meta["chunk"]
            chunk_id = str(meta["id"])

            try:
                results = ner_model.predict(text=chunk.strip())
                if not results:
                    logger.debug("No entities found for chunk_id %s", chunk_id)
                    continue

                inserted = insert_ner_entities(conn=conn,
                                              chunk_id=chunk_id,
                                              entities=results)
                if inserted:
                    entity_total += len(results)
                    logger.info("Inserted %d entities for chunk_id %s", len(results), chunk_id)
                else:
                    logger.warning("Failed to insert entities for chunk_id %s", chunk_id)

            except Exception as e:
                logger.error("❌ Failed to process NER for chunk ID %s: %s", chunk_id, e, exc_info=True)

        logger.info("✅ Inserted %d entities from %d chunks.", entity_total, len(valid_chunks))

        return JSONResponse(
            content={
                "message": "NER extraction and insertion completed successfully.",
                "chunks_processed": len(valid_chunks),
                "entities_inserted": entity_total
            },
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.exception("💥 Unexpected error in NER extraction pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during NER processing."
        ) from e

