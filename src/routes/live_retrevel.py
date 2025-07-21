"""
live_retrieval_route.py

This module provides a FastAPI endpoint for real-time semantic retrieval using the PathRAG engine.

It enables users to submit a query string and receive a contextually relevant prompt
generated from semantically related document chunks in a graph-based structure.

Route:
    POST /api/v1/retrieval

Query Parameters:
    - query (str): User's natural language question or search query.
    - top_k (int): Number of semantically closest nodes to retrieve from the graph (default: 3).
    - max_hop (int): Maximum number of hops to allow for exploring evidence paths (default: 2).

Core Logic:
    1. Retrieve the top-K most relevant nodes based on query embedding similarity.
    2. Identify and prune graph paths between those nodes using a maximum hop constraint.
    3. Score the paths to prioritize coherent and meaningful reasoning chains.
    4. Generate a prompt from the top-ranked paths that can be used for downstream QA or reasoning.

Raises:
    - HTTPException 500 if an error occurs during retrieval or prompt generation.

Dependencies:
    - PathRAG (semantic graph-based retrieval engine)
    - FastAPI
    - Custom helpers and dependency injection via `get_path_rag()`

Logging:
    - Logs all major steps including query receipt, retrieval progress, scoring, and prompt generation.

Author:
    ALRashid AlKiswane
"""

import os
import sys
import logging
from fastapi import (APIRouter,
                     HTTPException,
                     Depends,
                     Query
                     )

from fastapi.responses import JSONResponse

# Setup project base path for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical("Failed to configure project base directory: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.rag import  PathRAG
from src.infra import setup_logging
from src import get_path_rag

# === Logger Setup ===
logger = setup_logging(name="LIVE-RETREVAL")

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
