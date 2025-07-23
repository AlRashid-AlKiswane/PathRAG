"""
build_pathrag_route.py

This module defines the FastAPI route for building the PathRAG semantic graph used in
path-aware retrieval-augmented generation (RAG).

The route `/api/v1/build_pathrag` loads embedded document chunks from the MongoDB
`embed_vector` collection and constructs a semantic similarity graph using cosine similarity.
The graph enables multi-hop reasoning and structured evidence retrieval in PathRAG.

Main Functionality:
    - Loads and parses embedded chunks from MongoDB.
    - Validates and transforms embedding vectors.
    - Builds a graph using the PathRAG engine with edge weights based on vector similarity.
    - Returns the number of nodes and edges in the final graph.

Route:
    POST /api/v1/build_pathrag

Dependencies:
    - MongoDB connection (`get_mongo_db`)
    - PathRAG graph engine (`get_path_rag`)
    - FastAPI, NumPy, TQDM, and JSON libraries.

Author:
    ALRashid AlKiswane
"""

import json
import os
import sys
import logging
from typing import Optional

import numpy as np
from tqdm import tqdm
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to configure project root: %s", e)
    sys.exit(1)

# Internal imports (ensure MAIN_DIR is set correctly above)
from src.graph_db import pull_from_collection
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_mongo_db, get_path_rag
from src.rag import PathRAG

# Logger & Settings
logger = setup_logging(name="BUILD-PathRAG")
app_settings: Settings = get_settings()

# FastAPI router
build_pathrag_route = APIRouter(
    prefix="/api/v1/build_pathrag",
    tags=["Build PathRAG"],
    responses={404: {"description": "Not found"}}
)

@build_pathrag_route.post("", status_code=status.HTTP_201_CREATED)
async def build_pathrag(
    do_save: bool = False,
    build_graph: bool = False,
    limit: Optional[int] = Query(None, description="Optional limit on the number of chunks to load."),
    db: MongoClient = Depends(get_mongo_db),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Trigger the building of the PathRAG semantic graph using embedded text chunks from MongoDB.

    Args:
        limit (Optional[int]): Optional cap on number of embeddings to use.
        db (MongoClient): Injected MongoDB connection from FastAPI app state.
        pathrag (PathRAG): PathRAG engine instance with internal graph.

    Returns:
        JSONResponse: Success message with the number of nodes and edges built.
    """
    try:
        if build_graph:
            logger.info("Pulling embedded chunks from MongoDB collection 'embed_vector'...")
            rows = pull_from_collection(
                db=db,
                collection_name="embed_vector",
                fields=["chunk", "embedding"],
                limit=limit
            )

            if not rows:
                logger.warning("No documents found in 'embed_vector' collection.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No embedded chunks found in MongoDB."
                )

            chunks, embeddings = [], []

            for row in tqdm(rows, desc="Parsing Embeddings"):
                try:
                    chunk_text = row["chunk"].strip().replace("\n", " ")
                    embedding = json.loads(row["embedding"])
                    embedding_vector = np.array(embedding, dtype=np.float32)

                    chunks.append(chunk_text)
                    embeddings.append(embedding_vector)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning("Skipping malformed embedding row: %s", e)
                    continue

            if not chunks or not embeddings:
                logger.error("No valid chunks or embeddings parsed from MongoDB.")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid embeddings were found to construct the graph."
                )

            embeddings_np = np.vstack(embeddings)

            logger.info("Building semantic graph with %d valid chunks.", len(chunks))
            pathrag.build_graph(chunks=chunks, embeddings=embeddings_np)

            node_count = len(pathrag.g.nodes)
            edge_count = len(pathrag.g.edges)
            logger.info("Graph built successfully: %d nodes, %d edges", node_count, edge_count)

        if do_save:
            pathrag.save_graph(file_path=app_settings.STORGE_GRAPH)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Semantic graph built successfully.",
                "nodes": node_count,
                "edges": edge_count
            }
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logger.exception("Unexpected error while building semantic graph.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during graph construction."
        ) from e
