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
import numpy as np

# Setup project base path for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to configure project base directory: %s", e, exc_info=True)
    sys.exit(1)

# Project imports
from src.rag import FaissRAG, EntityLevelFiltering
from src.infra import setup_logging
from src import get_embedding_model, get_faiss_rag, get_entity_level_filtering
from src.llms_providers import HuggingFaceModel

logger = setup_logging()

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
        regex="^(intersection|union|faiss_only|entity_only)$"
    ),
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
    logger.info(f"Received retrieval request - Query: '{query}', Top_k: {top_k}, Mode: '{mode}'")

    if not query or not query.strip():
        logger.warning("Empty or whitespace-only query received.")
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        logger.debug("Generating embedding for query.")
        embed_query = embed_model.embed_texts(query)
        if not isinstance(embed_query, np.ndarray):
            embed_query = np.array(embed_query, dtype=np.float32)
        logger.debug(f"Embedding generated with shape: {embed_query.shape}")

    except Exception as embed_exc:
        logger.error(f"Embedding generation failed: {embed_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embeddings for the query."
        ) from embed_exc

    try:
        logger.debug("Performing semantic retrieval via FaissRAG.")
        results = faiss_rag.semantic_retrieval(embed_query=embed_query, top_k=top_k)
        logger.info(f"FaissRAG returned {len(results)} chunks.")

    except Exception as faiss_exc:
        logger.error(f"FaissRAG semantic retrieval failed: {faiss_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Semantic retrieval failed."
        ) from faiss_exc

    try:
        logger.debug("Performing entity-level retrieval.")
        entity_result = entity_level_filtering.entities_retrieval(query=query,
                                                                  top_k=top_k)
        logger.info(f"EntityLevelFiltering returned {len(entity_result)} chunks.")

    except Exception as entity_exc:
        logger.error(f"Entity-level retrieval failed: {entity_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Entity-level retrieval failed."
        ) from entity_exc

    try:
        # Extract sets of chunk identifiers
        faiss_chunks = set(chunk.get("chunk") or chunk.get("chunk_id") for chunk in results)
        entity_chunks = set(
            chunk.get("chunk") if isinstance(chunk, dict) else chunk
            for chunk in entity_result
        )

        logger.info(f"Combining results using mode: {mode}")

        if mode == "intersection":
            combined_chunks = faiss_chunks.intersection(entity_chunks)
            filtered_results = [
                chunk for chunk in results if (chunk.get("chunk") or chunk.get("chunk_id")) in combined_chunks
            ]

        elif mode == "union":
            combined_chunks = faiss_chunks.union(entity_chunks)
            # Map chunks to dict for fast lookup
            faiss_map = {chunk.get("chunk") or chunk.get("chunk_id"): chunk for chunk in results}
            entity_map = {}
            for ent_chunk in entity_result:
                key = ent_chunk.get("chunk") if isinstance(ent_chunk, dict) else ent_chunk
                entity_map[key] = ent_chunk if isinstance(ent_chunk, dict) else {"chunk": key}

            # Combine: Faiss results first, then entity results not in Faiss
            filtered_results = [faiss_map[c] for c in combined_chunks if c in faiss_map]
            filtered_results.extend([entity_map[c] for c in combined_chunks if c not in faiss_map])

        elif mode == "faiss_only":
            filtered_results = results

        elif mode == "entity_only":
            filtered_results = []
            for ent_chunk in entity_result:
                if isinstance(ent_chunk, dict):
                    filtered_results.append(ent_chunk)
                else:
                    filtered_results.append({"chunk": ent_chunk})

        else:
            # Defensive fallback (should not happen due to regex validation)
            logger.warning(f"Unknown mode '{mode}', defaulting to 'intersection'.")
            combined_chunks = faiss_chunks.intersection(entity_chunks)
            filtered_results = [
                chunk for chunk in results if (chunk.get("chunk") or chunk.get("chunk_id")) in combined_chunks
            ]

        logger.info(f"Returning {len(filtered_results)} chunks for mode '{mode}'.")

        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "top_k": top_k,
                "mode": mode,
                "results": filtered_results
            }
        )

    except Exception as combine_exc:
        logger.error(f"Failed to combine retrieval results: {combine_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to combine retrieval results."
        ) from combine_exc
