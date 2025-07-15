"""
Live Retrieval API Module

This module defines a FastAPI route that handles live semantic retrieval queries.
It uses two retrieval mechanisms:
1. FaissRAG: semantic vector similarity search
2. EntityLevelFiltering: entity-based chunk filtering

The API combines retrieval results according to a specified mode:
- intersection: common chunks only
- union: all chunks from both sources
- faiss_only: only FaissRAG results
- entity_only: only entity-level results

Key Features:
- Dependency injection for reusable components
- Embedding generation with HuggingFaceModel
- Flexible retrieval result combination modes
- Comprehensive logging for traceability and debugging
- Full error handling for robustness

Typical Usage:
    POST /api/v1/retrieval?query="What is AI?"&top_k=5&mode=intersection
"""

import os
import sys
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlite3 import Connection
# Setup project base path for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical("🚨 Failed to configure project base directory: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.rag import FaissRAG, EntityLevelFiltering, dual_level_retrieval
from src.infra import setup_logging
from src import (get_embedding_model,
                 get_faiss_rag,
                 get_entity_level_filtering,
                 get_db_conn)
from src.llms_providers import HuggingFaceModel

# === Logger Setup ===
logger = setup_logging()

# === FastAPI Router ===
live_retrieval_route = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Live Retrieval Query"],
    responses={404: {"description": "Not found"}}
)


@live_retrieval_route.post("", response_class=JSONResponse)
async def retrieve(
    query: str = Query(..., description="Query string to search relevant chunks"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top results to retrieve"),
    mode: str = Query(
        "intersection",
        description="Mode of combining results: 'intersection', 'union', 'faiss_only', 'entity_only'",
        pattern="^(intersection|union|faiss_only|entity_only)$"
    ),
    conn: Connection = Depends(get_db_conn),
    embed_model: HuggingFaceModel = Depends(get_embedding_model),
    faiss_rag: FaissRAG = Depends(get_faiss_rag),
    entity_level_filtering: EntityLevelFiltering = Depends(get_entity_level_filtering)
):
    """
    Handle live retrieval requests by querying FaissRAG and EntityLevelFiltering,
    and returning results combined according to the specified mode.

    Args:
        query (str): The user query string.
        top_k (int): Number of top chunks to retrieve.
        mode (str): Strategy to combine retrieval results.
            Options:
            - 'intersection': return common chunks only.
            - 'union': return all chunks from both sources.
            - 'faiss_only': return only FaissRAG results.
            - 'entity_only': return only entity-level results.
        embed_model (HuggingFaceModel): Embedding model injected by FastAPI.
        faiss_rag (FaissRAG): FaissRAG retrieval instance injected by FastAPI.
        entity_level_filtering (EntityLevelFiltering): Entity-level retrieval instance.

    Returns:
        JSONResponse: JSON with retrieved chunk data.

    Raises:
        HTTPException: For client errors (e.g., empty query) and server errors.
    """
    logger.info("📥 Received retrieval request | Query: '%s' | Top_k: %d | Mode: '%s'", query, top_k, mode)

    try:
        filtered_results = dual_level_retrieval(
            conn=conn,
            query=query,
            top_k=top_k,
            mode=mode,
            embed_model=embed_model,
            faiss_rag=faiss_rag,
            entity_level_filtering=entity_level_filtering
        )
        logger.info("📤 Returning %d chunks for mode '%s'.", len(filtered_results), mode)

        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "top_k": top_k,
                "mode": mode,
                "results": filtered_results
            }
        )

    except HTTPException as http_exc:
        logger.warning("⚠️ Retrieval failed due to user input: %s", http_exc.detail)
        raise http_exc

    except Exception as e:
        logger.error("🔥 Unexpected retrieval failure: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to perform dual-level retrieval."
        ) from e
