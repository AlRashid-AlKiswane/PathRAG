"""
main.py

This is the main entry point for the Graph-RAG FastAPI application, which implements
a Path-aware Retrieval-Augmented Generation system using semantic graphs.

Responsibilities:
- Configure project path and imports for internal modules
- Initialize application settings and logging
- Manage application lifespan including startup and graceful shutdown:
    - Setup SQLite database connection and tables
    - Load language models (OllamaModel and HuggingFaceModel)
    - Initialize PathRAG reasoning engine
- Register API routes in logical workflow order:
    1. File upload
    2. Chunking of documents
    3. Embedding generation for chunks
    4. Live retrieval of relevant information
    5. Chatbot interface for user interaction
- Handle critical errors during startup or route registration with logging and termination

This module aims for robustness, clarity, and maintainability with thorough error
handling and descriptive logging for each step.
"""

import json
import os
from pathlib import Path
import sys
import logging
from typing import Union
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
# === Configure Project Path ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("❌ Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Internal Imports ===
try:
    from src.graph_db import (
        get_mongo_client,
        init_chatbot_collection,
        init_chunks_collection,
        init_embed_vector_collection,
    )
    from src.routes import *
    from src.infra import setup_logging
    from src.llms_providers import OllamaModel, HuggingFaceModel
    from src.rag import PathRAG
    from src.helpers import get_settings, Settings
    from src.utils import sanitize
except ImportError as e:
    logging.critical("❌ Import error during module loading: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging(name="MAIN")
app_settings: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for app startup and shutdown.

    On startup:
    - Connect to MongoDB and initialize collections
    - Load language and embedding models
    - Set up the PathRAG reasoning system

    On shutdown:
    - MongoDB does not require manual closure

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None

    Raises:
        RuntimeError: On failure of any critical component.
    """
    try:
        logger.info("🚀 Starting up Graph-RAG API...")

        # === MongoDB Initialization ===
        try:
            client = get_mongo_client()
            db = client["PathRAG-MongoDB"]
            app.state.db = db
            logger.info("✅ Connected to MongoDB: %s", db.name)
        except Exception as mongo_err:
            logger.critical("❌ Failed to connect to MongoDB: %s", mongo_err, exc_info=True)
            raise RuntimeError("Database initialization failed.") from mongo_err

        # === Initialize MongoDB Collections ===
        try:
            init_chunks_collection(db)
            init_embed_vector_collection(db)
            init_chatbot_collection(db)
            logger.info("✅ MongoDB collections initialized.")
        except Exception as coll_err:
            logger.critical("❌ Failed to initialize MongoDB collections: %s", coll_err, exc_info=True)
            raise RuntimeError("Collection setup failed.") from coll_err

        # === Load Models ===
        try:
            app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
            app.state.embedding_model = HuggingFaceModel(app_settings.EMBEDDING_MODEL)
            logger.info("✅ LLM and embedding models loaded.")
        except Exception as model_err:
            logger.critical("❌ Failed to load models: %s", model_err, exc_info=True)
            raise RuntimeError("Model loading failed.") from model_err

        # === Initialize PathRAG ===
        try:
            global path_rag
            path_rag = PathRAG(
                embedding_model=app.state.embedding_model,
                decay_rate=app_settings.DECAY_RATE,
                prune_thresh=app_settings.PRUNE_THRESH,
                sim_should=app_settings.SIM_THRESHOLD
            )
            app.state.path_rag = path_rag
            graph_path = Path(app_settings.STORGE_GRAPH)
            if graph_path.exists():
                try:
                    path_rag.load_graph(graph_path)
                except Exception as e:
                    logger.error("Feailed to load graph: %s", e)
            
            else:
                logger.info("No existing graph found - will create new when needed")

            logger.info("✅ PathRAG initialized.")
        except Exception as rag_err:
            logger.critical("❌ Failed to initialize PathRAG: %s", rag_err, exc_info=True)
            raise RuntimeError("PathRAG initialization failed.") from rag_err

        yield  # Application runs here

    except Exception as startup_error:
        logger.exception("🔥 Fatal error during app startup: %s", startup_error)
        raise

    finally:
        logger.info("🛑 Application shutdown complete.")


# === FastAPI Application Instance ===
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="Path-aware Retrieval-Augmented Generation system using semantic graphs.",
    lifespan=lifespan
)

# Serve UI HTML
@app.get("/")
async def serve_ui():
    return FileResponse(f"{MAIN_DIR}/src/web/index.html")

# Serve static assets if needed
app.mount("/static", StaticFiles(directory=f"{MAIN_DIR}/src/web"), name="static")

@app.get("/graph")
async def get_graph(max_nodes: int = 100):
    """
    Returns a Plotly graph JSON representation of the semantic graph.

    Args:
        max_nodes (int): Maximum number of nodes to visualize.

    Returns:
        JSONResponse: Plotly figure as JSON.
    """
    try:
        g = path_rag.g
        if g is None or g.number_of_nodes() == 0:
            logging.warning("Graph is empty or not initialized.")
            raise HTTPException(status_code=404, detail="Graph is empty or not initialized.")

        logging.info(f"Visualizing graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
        fig = path_rag.visualize_graph(max_nodes=max_nodes)
        return JSONResponse(content=json.loads(fig.to_json()))

    except Exception as e:
        logging.exception("Failed to generate graph visualization.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graph-data")
async def get_graph_data():
    """
    Returns graph node data as a list of records.

    Returns:
        List[Dict]: List of node dictionaries.
    """
    try:
        df = path_rag.to_dataframe()
        if df.empty:
            logging.warning("Graph dataframe is empty.")
            raise HTTPException(status_code=404, detail="No node data available.")

        return df.to_dict(orient="records")

    except Exception as e:
        logging.exception("Failed to retrieve graph data.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graph/node/{node_id}")
async def get_node_info(node_id: str):
    """
    Returns detailed information for a given graph node.

    Args:
        node_id (str): Node identifier (can be string or integer).

    Returns:
        Dict: Node metadata and text content.
    """
    try:
        g = path_rag.g
        if g is None or g.number_of_nodes() == 0:
            logging.warning("Graph is not available.")
            raise HTTPException(status_code=404, detail="Graph is empty or not initialized.")

        possible_ids: list[Union[str, int]] = [node_id]
        if node_id.isdigit():
            possible_ids.append(int(node_id))

        for pid in possible_ids:
            if pid in g.nodes:
                raw_node = g.nodes[pid]
                node_data = dict(raw_node)

                return {
                    "node_id": str(pid),
                    "label": node_data.get("label", ""),
                    "text": node_data.get("text", ""),
                    "metadata": sanitize(node_data)  # Safe for JSON
                }

        logging.warning(f"Node ID '{node_id}' not found in graph.")
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    except Exception as e:
        logging.exception(f"Error retrieving node '{node_id}'.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Register Routes ===
try:
    app.include_router(upload_route)
    app.include_router(chunking_route)
    app.include_router(embedding_chunks_route)
    app.include_router(build_pathrag_route)
    app.include_router(live_retrieval_route)
    app.include_router(chatbot_route)
    app.include_router(storage_management_route)
    app.include_router(resource_monitor_router)
    app.include_router(md_chunker_routes)
    logger.info("✅ All API routes registered successfully.")
except Exception as route_err:
    logger.critical("❌ Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)
