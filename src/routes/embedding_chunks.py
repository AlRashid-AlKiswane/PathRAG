"""
Chunks to Embedding Vectors API - FastAPI Module

This module defines a FastAPI POST endpoint that retrieves chunked text data from
a specified SQLite table and generates embeddings using a HuggingFace model. The
embeddings are stored in the 'embed_vector' table for use in a graph-based RAG system.

Features:
- Dependency injection for SQLite connection and embedding model (HuggingFaceModel).
- Flexible chunk extraction based on table and columns.
- Embedding generation and insertion into the 'embed_vector' table.
- Full-level error handling and structured logging for traceability.

Raises:
- HTTP 500 for database or processing errors.
- HTTP 504 for embedding timeouts.
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
from src.llms_providers import HuggingFaceModel
from src.db import pull_from_table, insert_embed_vector
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_db_conn, get_embedding_model, get_faiss_rag
from src.rag import FaissRAG

# === Logger & Settings ===
logger = setup_logging()
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
    faiss_rag: FaissRAG = Depends(get_faiss_rag)
) -> JSONResponse:
    """
    Retrieve text chunks from a database, generate embeddings,
    and store them in the 'embed_vector' table.

    Args:
        columns (List[str]): Columns to retrieve. Default: ["id", "chunk", "dataName"].
        table_name (str): Source table for chunks.
        conn (Connection): SQLite DB connection.
        embedding_model (HuggingFaceModel): Embedding model instance.
        faiss_rag (FaissRAG): FAISS index manager.

    Returns:
        JSONResponse: Count of successfully embedded and stored chunks.
    """
    try:
        logger.info("📦 Retrieving chunks from table '%s' with columns: %s", table_name, columns)

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

        logger.debug("🧩 %d valid chunks retrieved. Sample: %s...", len(valid_chunks), valid_chunks[0]['chunk'][:100])

        processed_count = 0

        for meta in valid_chunks:
            chunk = meta["chunk"]
            chunk_id = meta["id"]

            try:
                logger.debug("🔄 Embedding chunk ID: %s | Text: %.50s...", chunk_id, chunk)

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
                else:
                    logger.warning("⚠️ Insertion failed for chunk ID %s", chunk_id)

            except Exception as embed_err:
                logger.error("💥 Embedding error on chunk ID %s: %s", chunk_id, embed_err, exc_info=True)

        logger.info("✅ %d/%d chunks embedded and stored.", processed_count, len(valid_chunks))

        # 🔄 Refresh FAISS internal state
        faiss_rag.vectors_embedding, faiss_rag.chunk_ids = faiss_rag._fetch_embedding_vectors()
        faiss_rag.index = faiss_rag._build_faiss_index()

        return JSONResponse(
            content={
                "message": "Chunks successfully embedded and stored.",
                "chunks_processed": processed_count
            },
            status_code=status.HTTP_200_OK
        )

    # --- Error Handling ---
    except (OperationalError, DatabaseError) as db_err:
        logger.exception("❌ Database error during chunk processing.")
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
        logger.error("🚫 Embedding process was cancelled.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding operation was cancelled."
        )

    except HTTPException as http_err:
        logger.warning("⚠️ HTTPException: %s", http_err.detail)
        raise http_err

    except Exception as e:
        logger.exception("💣 Unexpected error in embedding pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during chunk embedding."
        ) from e
