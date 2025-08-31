#!/usr/bin/env python3
"""
build_pathrag_route.py

FastAPI route to build the PathRAG semantic graph from MongoDB embedded chunks.
Provides multi-hop reasoning for retrieval-augmented generation (RAG).

Author: ALRashid AlKiswane
"""

import json
import os
import sys
import time
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

# Internal imports
from src.mongodb import pull_from_collection
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_mongo_db, get_path_rag
from src.rag import PathRAG
from src.utils import AutoSave

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
    method: str = "knn",
    limit: Optional[int] = Query(None, description="Optional limit on number of chunks to load."),
    db: MongoClient = Depends(get_mongo_db),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Build the PathRAG semantic graph using embedded chunks from MongoDB.

    Args:
        do_save (bool): Whether to save graph and checkpoint after building.
        build_graph (bool): Whether to trigger graph building.
        limit (Optional[int]): Max number of chunks to load from MongoDB.
        db (MongoClient): MongoDB connection injected by FastAPI.
        pathrag (PathRAG): PathRAG engine instance.

    Returns:
        JSONResponse: Success message with nodes and edges counts.
    """
    try:
        if not build_graph:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Graph building not triggered."}
            )

        logger.info("Pulling embedded chunks from MongoDB 'embed_vector' collection...")
        rows = pull_from_collection(
            db=db,
            collection_name="embed_vector",
            fields=["chunk", "embedding"],
            limit=limit
        )

        if not rows:
            logger.warning("No documents found in 'embed_vector'.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No embedded chunks found in MongoDB."
            )

        chunks, embeddings = [], []

        for row in tqdm(rows, desc="Parsing embeddings"):
            try:
                chunk_text = row["chunk"].strip().replace("\n", " ")
                embedding_vector = np.array(json.loads(row["embedding"]), dtype=np.float32)
                chunks.append(chunk_text)
                embeddings.append(embedding_vector)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning("Skipping malformed embedding row: %s", e)
                continue

        if not chunks or not embeddings:
            logger.error("No valid chunks or embeddings parsed.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid embeddings found to build the graph."
            )

        embeddings_np = np.vstack(embeddings)
        logger.info("Building semantic graph with %d valid chunks.", len(chunks))
        start_time = time.time()
        pathrag.build_graph(chunks=chunks, embeddings=embeddings_np, method=method)
        build_time = time.time() - start_time
        logger.info("Graph built in %.2f seconds.", build_time)

        # Metrics
        metrics = pathrag.get_metrics()
        logger.info("Graph Statistics: Nodes=%d, Edges=%d, Memory=%.2f GB, AvgDegree=%.2f",
                    metrics["nodes_count"], metrics["edges_count"],
                    metrics["memory_usage_gb"], metrics["avg_degree"])

        # Save checkpoint and graph if requested
        if do_save:
            autosave = AutoSave(pathrag_instance=pathrag)
            autosave.save_checkpoint()

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Semantic graph built successfully.",
                "metrics": metrics
            }
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception:
        logger.exception("Unexpected error during PathRAG graph building.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during graph construction."
        )
