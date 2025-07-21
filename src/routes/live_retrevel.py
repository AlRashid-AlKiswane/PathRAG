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
from src.rag import  PathRAG
from src.infra import setup_logging
from src import get_path_rag

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
    query: str = Query(..., description="Query string to search relevant chunks."),
    top_k: int = Query(3, ge=1, le=10, description="Number of top similar chunks to retrieve."),
    max_hop: int = Query(2, ge=1, le=10, description="Maximum number of hops to allow in path search."),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Perform a live semantic retrieval query using PathRAG.

    This endpoint retrieves semantically relevant document chunks from a pre-built
    semantic graph using a path-aware approach, then generates a context-rich prompt.

    Args:
        query (str): The user's query string.
        top_k (int): Number of top similar nodes to retrieve initially.
        max_hop (int): Max number of hops allowed when searching paths.
        pathrag (PathRAG): The PathRAG engine injected from app state.

    Returns:
        JSONResponse: Contains the generated prompt and supporting path metadata.

    Raises:
        HTTPException: If an internal error occurs during retrieval.
    """
    try:
        logger.info("Received retrieval query: '%s'", query)

        # Step 1: Retrieve top-K nodes semantically similar to query
        logger.debug("Retrieving top-%d similar nodes...", top_k)
        nodes = pathrag.retrieve_nodes(query=query, top_k=top_k)

        # Step 2: Prune and validate semantic paths
        logger.debug("Pruning paths with max_hop=%d...", max_hop)
        paths = pathrag.prune_paths(nodes=nodes, max_hops=max_hop)

        if not paths:
            logger.warning("No valid paths found for query: %s", query)
            return JSONResponse(
                status_code=200,
                content={"message": "[!] No valid paths found. Try lowering prune_thresh or increasing max_hops."}
            )

        # Step 3: Score and rank the retrieved paths
        logger.debug("Scoring %d candidate paths...", len(paths))
        scored_paths = pathrag.score_paths(paths=paths)

        # Step 4: Generate final prompt
        logger.debug("Generating prompt using scored paths.")
        prompt = pathrag.generate_prompt(query=query, scored_paths=scored_paths)

        logger.info("Prompt successfully generated for query.")

        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "top_k": top_k,
                "max_hop": max_hop,
                "num_paths": len(scored_paths),
                "prompt": prompt
            }
        )

    except Exception as e:
        logger.exception("Failed to complete retrieval for query: %s", query)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing semantic retrieval query."
        ) from e
