"""
embedding_chunks_route.py

This module defines the FastAPI route for converting textual document chunks
into vector embeddings and storing them in a database table (`embed_vector`).

It uses a local HuggingFace embedding model to generate dense vector representations
of preprocessed document segments retrieved from a SQLite database.

Route:
    POST /api/v1/chunks_to_embeddings/

Query Parameters (defaults provided in code):
    - columns (List[str]): Columns to fetch from the source table (default: ["id", "chunk", "dataName"])
    - table_name (str): Name of the source table to retrieve chunks from (default: "chunks")

Main Workflow:
    1. Retrieve chunks (text segments) from the specified table and columns.
    2. Filter out invalid or empty chunks.
    3. Generate embeddings using a HuggingFace model.
    4. Serialize the embedding vectors and insert them into the `embed_vector` table.
    5. Return a count of successfully processed chunks.

Raises:
    - 404 if no valid chunks are found.
    - 500 for database errors or embedding/model failures.
    - 504 if the embedding process times out.

Dependencies:
    - FastAPI
    - HuggingFaceModel (custom embedding provider)
    - SQLite (via Python’s built-in sqlite3 module)
    - Custom DB logic from `insert_embed_vector` and `pull_from_table`

Logging:
    - Logs all major steps and errors with contextual information.

Author:
    ALRashid AlKiswane
"""

import asyncio
import os
import sys
import logging
import json
from typing import List
from sqlite3 import (Connection,
                     OperationalError,
                     DatabaseError,
                     )

from fastapi import (APIRouter,
                     HTTPException,
                     status,
                     Depends,
                     )

from fastapi.responses import JSONResponse
from tqdm import tqdm

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.llms_providers import HuggingFaceModel
from src.db import pull_from_table, insert_embed_vector
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_db_conn, get_embedding_model

# === Logger & Settings ===
logger = setup_logging(name="EMEBDDING-CHUNKS")
app_settings: Settings = get_settings()

# === API Router ===
embedding_chunks_route = APIRouter(
    prefix="/api/v1/chunks_to_embeddings",
    tags=["Chunks → Embeddings"],
    responses={404: {"description": "Not found"}}
)


@embedding_chunks_route.post("", response_class=JSONResponse)
async def chunks_to_embeddings(
    columns: List[str] = ["id", "chunk", "dataName"],
    table_name: str = "chunks",
    conn: Connection = Depends(get_db_conn),
    embedding_model: HuggingFaceModel = Depends(get_embedding_model),
) -> JSONResponse:
    """
    Retrieve text chunks from a database, generate embeddings,
    and store them in the 'embed_vector' table.

    Args:
        columns (List[str]): Columns to retrieve. Default: ["id", "chunk", "dataName"].
        table_name (str): Source table for chunks.
        conn (Connection): SQLite DB connection.
        embedding_model (HuggingFaceModel): Embedding model instance.

    Returns:
        JSONResponse: Count of successfully embedded and stored chunks.
    """
    try:
        logger.info("Retrieving chunks from table '%s' with columns: %s", table_name, columns)

        meta_chunks = pull_from_table(conn=conn, columns=columns, table_name=table_name)
        if not meta_chunks:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No chunks found in table '{table_name}'."
            )

        valid_chunks = [row for row in meta_chunks if row.get("chunk")]
        if not valid_chunks:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No usable chunk data found."
            )

        logger.debug("%d valid chunks retrieved. Sample: %s...", len(valid_chunks), valid_chunks[0]['chunk'][:100])

        processed_count = 0

        # Wrap loop with tqdm to show a progress bar
        for meta in tqdm(valid_chunks, desc="Embedding Chunks", unit="chunk"):
            chunk = meta["chunk"]
            chunk_id = meta["id"]

            try:
                embedding_vector = embedding_model.embed_texts(
                    texts=chunk,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                if hasattr(embedding_vector, "tolist"):
                    embedding_vector = embedding_vector.tolist()

                serialized = json.dumps(embedding_vector)

                success = insert_embed_vector(
                    conn=conn,
                    chunk=chunk,
                    embedding=serialized,
                    chunk_id=str(chunk_id)
                )

                if success:
                    processed_count += 1

            except Exception as embed_err:
                tqdm.write(f"[ERROR] Embedding failed on chunk ID {chunk_id}: {embed_err}")
        logger.info("✅ %d/%d chunks embedded and stored.", processed_count, len(valid_chunks))

        return JSONResponse(
            content={
                "message": "Chunks successfully embedded and stored.",
                "chunks_processed": processed_count
            },
            status_code=status.HTTP_200_OK
        )

    # --- Error Handling ---
    except (OperationalError, DatabaseError) as db_err:
        logger.exception("Database error during chunk processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during embedding process."
        ) from db_err

    except asyncio.TimeoutError:
        logger.error("⌛ Embedding operation timed out.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding generation timed out."
        )

    except asyncio.CancelledError:
        logger.error("Embedding process was cancelled.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding operation was cancelled."
        )

    except HTTPException as http_err:
        logger.warning("HTTPException: %s", http_err.detail)
        raise http_err

    except Exception as e:
        logger.exception("💣 Unexpected error in embedding pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during chunk embedding."
        ) from e
