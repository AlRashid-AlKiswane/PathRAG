"""
Live Retrieval API

This module defines a FastAPI route that handles live semantic queries using
a FaissRAG instance. It allows users to pass in a question and receive
semantically similar chunks of information from the database.

Key Features:
- Accepts a user query and retrieves top-k relevant results
- Uses dependency injection for FaissRAG instance
- Provides clean logging and full error handling

Typical Usage:
    GET /api/v1/retrieval?query="What is AI?"&top_k=5
"""

import os
import sys
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np

# Set project base path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to configure project base directory: %s", e, exc_info=True)
    sys.exit(1)

# Project imports
from src.rag import FaissRAG
from src.infra import setup_logging
from src import get_embedding_model, get_faiss_rag
from src.llms_providers import HuggingFaceModel

logger = setup_logging()

live_retrieval_route = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Live Retrieval Query"],
    responses={404: {"description": "Not found"}}
)


@live_retrieval_route.get("", response_class=JSONResponse)
async def retrieve(
    query: str = Query(..., description="Query string to search relevant chunks"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top results to retrieve"),
    embed_model: HuggingFaceModel = Depends(get_embedding_model),
    faiss_rag: FaissRAG = Depends(get_faiss_rag)
):
    """
    Retrieves top-k relevant information chunks for a given query using FaissRAG.

    Args:
        query (str): The user-provided semantic question or search input.
        top_k (int): Number of top matching chunks to retrieve (default: 3).
        embed_model: Dependency-injected embedding model.
        faiss_rag (FaissRAG): Dependency-injected retriever.

    Returns:
        JSONResponse: Retrieved document chunks.

    Raises:
        HTTPException: For client or internal errors.
    """
    try:
        logger.info(f"Received retrieval request: query='{query}', top_k={top_k}")

        if not query.strip():
            logger.warning("Empty query string received.")
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        embed_query = embed_model.embed_texts(query)

        # Ensure it's a NumPy array
        if not isinstance(embed_query, np.ndarray):
            embed_query = np.array(embed_query, dtype=np.float32)

        logger.debug(f"Query embedding generated with shape: {embed_query.shape}")

        results = faiss_rag.semantic_retrieval(embed_query=embed_query, top_k=top_k)

        logger.info(f"Successfully retrieved {len(results)} chunks for query.")

        return JSONResponse(
            content={
                "query": query,
                "top_k": top_k,
                "results": results
            },
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Live retrieval failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during retrieval."
        ) from e
