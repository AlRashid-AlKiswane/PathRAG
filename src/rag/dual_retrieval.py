"""
Dual-Level Retrieval Module

This module defines the `dual_level_retrieval` function, which performs two-stage 
semantic retrieval using both vector similarity (via FAISS) and named entity-level 
filtering. It supports retrieval modes like 'intersection', 'union', and each 
individual retrieval path.

Features:
- Embedding-based semantic search
- Named entity filtering
- Result combination via configurable mode
- Full logging and error handling for robust diagnostics

Author: [Your Name]
Date: 2025-07-09
"""

import logging
from typing import List, Dict, Any
import numpy as np
from fastapi import HTTPException

# Assume these are implemented/imported correctly
from src.llms_providers import HuggingFaceModel
from src.rag import FaissRAG, EntityLevelFiltering

logger = logging.getLogger(__name__)


def dual_level_retrieval(
    query: str,
    top_k: int,
    mode: str,
    embed_model: HuggingFaceModel,
    faiss_rag: FaissRAG,
    entity_level_filtering: EntityLevelFiltering
) -> List[Dict[str, Any]]:
    """
    Perform dual-level semantic retrieval using FAISS and named entity filtering.

    Args:
        query (str): The search query text.
        top_k (int): Number of top results to retrieve.
        mode (str): Retrieval combination mode. One of:
                    - 'intersection'
                    - 'union'
                    - 'faiss_only'
                    - 'entity_only'
        embed_model: An object with an `.embed_texts()` method returning a vector.
        faiss_rag: An object with a `.semantic_retrieval()` method.
        entity_level_filtering: An object with `.entities_retrieval()` method.

    Returns:
        List[Dict[str, Any]]: Filtered list of retrieved chunks.

    Raises:
        HTTPException: On input validation or processing failure.
    """
    if not query or not query.strip():
        logger.warning("❗ Empty or whitespace-only query received.")
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # === Generate Embedding ===
    try:
        logger.debug("🔍 Generating embedding for query.")
        embed_query = embed_model.embed_texts(query)
        if not isinstance(embed_query, np.ndarray):
            embed_query = np.array(embed_query, dtype=np.float32)
        logger.debug("✅ Embedding generated. Shape: %s", embed_query.shape)
    except Exception as embed_exc:
        logger.error("❌ Embedding generation failed: %s", embed_exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embeddings for the query."
        ) from embed_exc

    # === FAISS Retrieval ===
    try:
        logger.debug("🔎 Performing semantic retrieval via FaissRAG.")
        results = faiss_rag.semantic_retrieval(embed_query=embed_query, top_k=top_k)
        logger.info("✅ FaissRAG returned %d chunks.", len(results))
    except Exception as faiss_exc:
        logger.error("❌ FAISS semantic retrieval failed: %s", faiss_exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Semantic retrieval failed."
        ) from faiss_exc

    # === Entity-Level Retrieval ===
    try:
        logger.debug("🔎 Performing entity-level retrieval.")
        entity_result = entity_level_filtering.entities_retrieval(query=query, top_k=top_k)
        logger.info("✅ EntityLevelFiltering returned %d chunks.", len(entity_result))
    except Exception as entity_exc:
        logger.error("❌ Entity-level retrieval failed: %s", entity_exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Entity-level retrieval failed."
        ) from entity_exc

    # === Combine Results ===
    try:
        faiss_chunks = set(chunk.get("chunk") or chunk.get("chunk_id") for chunk in results)
        entity_chunks = set(
            chunk.get("chunk") if isinstance(chunk, dict) else chunk
            for chunk in entity_result
        )

        logger.info("🔁 Combining results using mode: '%s'", mode)

        if mode == "intersection":
            combined_chunks = faiss_chunks.intersection(entity_chunks)
            filtered_results = [
                chunk for chunk in results if (chunk.get("chunk") or chunk.get("chunk_id")) in combined_chunks
            ]

        elif mode == "union":
            combined_chunks = faiss_chunks.union(entity_chunks)

            faiss_map = {chunk.get("chunk") or chunk.get("chunk_id"): chunk for chunk in results}
            entity_map = {
                (chunk.get("chunk") if isinstance(chunk, dict) else chunk):
                    (chunk if isinstance(chunk, dict) else {"chunk": chunk})
                for chunk in entity_result
            }

            filtered_results = [faiss_map[c] for c in combined_chunks if c in faiss_map]
            filtered_results.extend([entity_map[c] for c in combined_chunks if c not in faiss_map])

        elif mode == "faiss_only":
            filtered_results = results

        elif mode == "entity_only":
            filtered_results = [
                chunk if isinstance(chunk, dict) else {"chunk": chunk}
                for chunk in entity_result
            ]

        else:
            logger.warning("⚠️ Unknown mode '%s'. Defaulting to 'intersection'.", mode)
            combined_chunks = faiss_chunks.intersection(entity_chunks)
            filtered_results = [
                chunk for chunk in results if (chunk.get("chunk") or chunk.get("chunk_id")) in combined_chunks
            ]

        logger.info("✅ Final combined results: %d chunks", len(filtered_results))
        return filtered_results

    except Exception as combine_exc:
        logger.error("❌ Failed to combine retrieval results: %s", combine_exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to combine retrieval results."
        ) from combine_exc
