import json
import os
import sys
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlite3 import Connection

import numpy as np
from tqdm import tqdm

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.db import pull_from_table
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import (get_db_conn,
                 get_path_rag)

from src.rag import PathRAG 

# Initialize logger and settings
logger = setup_logging()
app_settings: Settings = get_settings()

build_pathrag_route = APIRouter(
    prefix="/api/v1/build_pathrag",
    tags=["Build PathRAG"],
    responses={404: {"description": "Not found"}}
)

@build_pathrag_route.post("", status_code=status.HTTP_201_CREATED)
async def build_pathrag(
    conn: Connection = Depends(get_db_conn),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Build the semantic graph used in PathRAG from all available embedded chunks in the database.

    This route pulls all document chunks and their embeddings from the database,
    processes them, and uses them to build the graph that enables path-aware
    retrieval and reasoning.

    Args:
        conn (Connection): SQLite database connection injected by FastAPI.
        pathrag (PathRAG): Initialized PathRAG instance from app state.

    Returns:
        JSONResponse: A success message with total nodes and edges, or a detailed error.
    """
    try:
        logger.info("Fetching embedded chunks from database...")
        rows = pull_from_table(conn=conn, table_name="embed_vector", columns=["chunk", "embedding"], limit=None)

        if not rows:
            logger.warning("No data found in 'embed_vector' table.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No embedded chunks found in the database."
            )

        chunks, embeddings = [], []
        for row in tqdm(rows, desc="Parsing Chunks & Embeddings"):
            try:
                chunk_text = row["chunk"].strip().replace("\n", " ")
                chunk_embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)

                chunks.append(chunk_text)
                embeddings.append(chunk_embedding)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning("Skipping row due to embedding error: %s", e)
                continue

        if not chunks or not embeddings:
            logger.error("No valid chunks or embeddings to build graph.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid chunks or embeddings found for graph construction."
            )

        embeddings = np.vstack(embeddings)

        logger.info("Building semantic graph with %d chunks...", len(chunks))
        pathrag.build_graph(chunks, embeddings)
        node_count = len(pathrag.g.nodes)
        edge_count = len(pathrag.g.edges)
        logger.info("Graph built: %d nodes, %d edges.", node_count, edge_count)

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
        logger.exception("Unexpected error while building PathRAG graph.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while building the PathRAG semantic graph."
        ) from e

