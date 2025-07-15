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
import os
import sys
from typing import List, Dict, Any
import numpy as np
from fastapi import HTTPException
from sqlite3 import Connection

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.llms_providers import HuggingFaceModel
from src.rag import FaissRAG, EntityLevelFiltering
from src.infra import setup_logging

logger = setup_logging()

def dual_level_retrieval(
    query: str,
    top_k: int,
    mode: str,
    conn: Connection,
    embed_model: HuggingFaceModel,
    faiss_rag: FaissRAG,
    entity_level_filtering: EntityLevelFiltering
) -> List[Dict[str, Any]]:
    """
    Perform dual-level semantic retrieval using FAISS and named entity filtering,
    and return full chunks.

    Args:
        query (str): The search query text.
        top_k (int): Number of top results to retrieve.
        mode (str): Retrieval combination mode.
        conn (Connection): SQLite connection.
        embed_model: Embedding model.
        faiss_rag: FAISS retrieval object.
        entity_level_filtering: NER-based retrieval object.

    Returns:
        List[Dict[str, Any]]: Full chunks retrieved.
    """
    if not query or not query.strip():
        logger.warning("❗ Empty or whitespace-only query received.")
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # === Generate embedding ===
    try:
        embed_query = embed_model.embed_texts(query)
        if not isinstance(embed_query, np.ndarray):
            embed_query = np.array(embed_query, dtype=np.float32)
    except Exception as e:
        logger.error("Embedding generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from e

    # === FAISS semantic retrieval ===
    try:
        faiss_results = faiss_rag.semantic_retrieval(embed_query=embed_query, top_k=top_k)
        faiss_chunk_map = {
            chunk.get("chunk_id") or chunk.get("chunk"): chunk for chunk in faiss_results
        }
        faiss_chunk_ids = set(faiss_chunk_map.keys())
    except Exception as e:
        logger.error("FAISS retrieval failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Semantic retrieval failed.") from e

    # === Entity-level retrieval ===
    try:
        entity_chunk_ids = entity_level_filtering.entities_retrieval(query=query, top_k=top_k)
        entity_chunk_ids_set = set(entity_chunk_ids)
    except Exception as e:
        logger.error("Entity-level retrieval failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Entity-level retrieval failed.") from e

    # === Fetch full chunks from DB for entity-only mode or union ===
    def fetch_chunks_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        try:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(ids))
            cursor.execute(f"SELECT * FROM embed_vector WHERE chunk_id IN ({placeholders})", ids)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return {row[0]: dict(zip(columns, row)) for row in rows}
        except Exception as e:
            logger.error("Failed to fetch full chunks from DB", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load chunks from DB.") from e

    # === Combine Results ===
    try:
        logger.info("Combining retrievals with mode: %s", mode)

        if mode == "intersection":
            combined_ids = faiss_chunk_ids.intersection(entity_chunk_ids_set)
            results = [faiss_chunk_map[cid] for cid in combined_ids if cid in faiss_chunk_map]

        elif mode == "union":
            combined_ids = faiss_chunk_ids.union(entity_chunk_ids_set)

            # Get any entity-only chunks not in faiss
            missing_ids = list(combined_ids - faiss_chunk_ids)
            extra_chunks = fetch_chunks_by_ids(missing_ids)
            results = [
                faiss_chunk_map[cid] if cid in faiss_chunk_map else extra_chunks[cid]
                for cid in combined_ids
                if cid in faiss_chunk_map or cid in extra_chunks
            ]

        elif mode == "faiss_only":
            results = list(faiss_chunk_map.values())

        elif mode == "entity_only":
            entity_chunks = fetch_chunks_by_ids(list(entity_chunk_ids_set))
            results = list(entity_chunks.values())

        else:
            logger.warning("Unknown mode '%s'. Defaulting to 'intersection'.", mode)
            combined_ids = faiss_chunk_ids.intersection(entity_chunk_ids_set)
            results = [faiss_chunk_map[cid] for cid in combined_ids if cid in faiss_chunk_map]

        logger.info("✅ Final combined results: %d chunks", len(results))
        return results

    except Exception as e:
        logger.error("Failed during result combination", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to combine retrieval results.") from e
