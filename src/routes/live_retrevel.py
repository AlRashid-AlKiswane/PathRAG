"""
Live Retrieval API

This module defines a FastAPI route that handles live semantic queries using
a LightRAG instance. It allows users to pass in a question and receive
semantically similar chunks of information from the knowledge graph or database.

Key Features:
- Accepts a user query and retrieves top-k relevant results
- Uses dependency injection for LightRAG instance
- Provides clean logging and error handling

Typical Usage:
    GET /api/v1/retrieval?query="What is AI?"&top_k=5
"""

import os
import sys
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to configure project base directory: %s", e, exc_info=True)
    sys.exit(1)

# Local imports
from src.rag import LightRAG
from src.infra import setup_logging
from src import get_light_rag

# Initialize logger
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
    light_rag: LightRAG = Depends(get_light_rag),
):
    """
    Retrieves top-k relevant information chunks for a given query.

    Args:
        query (str): The user-provided semantic question or search input.
        top_k (int): Number of top matching chunks to retrieve (default: 3).
        light_rag (LightRAG): Dependency-injected LightRAG instance.

    Returns:
        JSONResponse: JSON object containing retrieved chunks and their scores.

    Raises:
        HTTPException: 400 if query is missing, 500 for internal server errors.
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        results = light_rag.query(question=query, top_k=top_k)

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
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")
