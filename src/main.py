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

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

# === Configure Project Path ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug(f"Project base path configured: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Internal Imports ===
try:
    from src.db import (
        get_sqlite_engine,
        init_chunks_table,
        init_embed_vector_table,
        init_chatbot_table
    )
    from src.routes import (
        upload_route,
        chunking_route,
        embedding_chunks_route,
        live_retrieval_route,
        chatbot_route,
        storage_management_route,
        resource_monitor_router,
        build_pathrag_route
    )
    from src.infra import setup_logging
    from src.llms_providers import OllamaModel, HuggingFaceModel
    from src.rag import PathRAG
    from src.helpers import get_settings, Settings
except ImportError as e:
    logging.critical("Failed to import internal modules: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging(name="MAIN")
app_settings: Settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for application startup and shutdown.

    On startup:
    - Establish SQLite database connection
    - Initialize required tables
    - Load language and embedding models
    - Initialize PathRAG system

    On shutdown:
    - Close SQLite connection gracefully

    Args:
        app (FastAPI): The FastAPI application instance.

    Raises:
        ConnectionError: If database connection cannot be established.
        Exception: For any critical startup error to prevent app launch.
    """
    conn = None
    try:
        logger.info("Starting up Graph-RAG API.")

        # Initialize SQLite connection
        conn = get_sqlite_engine()
        if not conn:
            logger.error("Failed to get a valid SQLite connection.")
            raise ConnectionError("Failed to get a valid SQLite connection.")
        app.state.conn = conn
        logger.info("SQLite database connection established.")

        # Initialize required tables
        init_chunks_table(conn)
        init_embed_vector_table(conn)
        init_chatbot_table(conn)
        logger.info("Database tables initialized successfully.")

        # Load LLM and embedding models
        try:
            app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
            app.state.embedding_model = HuggingFaceModel(app_settings.EMBEDDING_MODEL)
            logger.info("Language and embedding models loaded successfully.")
        except Exception as model_err:
            logger.critical("Failed to load models: %s", model_err, exc_info=True)
            raise

        # Initialize PathRAG
        try:
            path_rag = PathRAG(
                embedding_model=app.state.embedding_model,
                decay_rate=app_settings.DECAY_RATE,
                prune_thresh=app_settings.PRUNE_THRESH,
                sim_should=app_settings.SIM_THRESHOLD
            )
            app.state.path_rag = path_rag
            logger.info("PathRAG initialized successfully.")
        except Exception as rag_err:
            logger.critical("Failed to initialize PathRAG: %s", rag_err, exc_info=True)
            raise

        yield

    except Exception as startup_error:
        logger.exception("Fatal error during FastAPI startup: %s", startup_error)
        raise

    finally:
        # Graceful shutdown
        if conn:
            try:
                conn.close()
                logger.info("SQLite connection closed.")
            except Exception as close_err:
                logger.warning("Error closing SQLite connection: %s", close_err, exc_info=True)


# === FastAPI Application Instance ===
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="A Path-aware Retrieval-Augmented Generation system using semantic graphs.",
    lifespan=lifespan
)

# === Register API Routes ===
try:
    # Register routes in logical workflow order:
    # Upload → Chunking → Embedding → Retrieval → Chatbot

    app.include_router(upload_route)
    app.include_router(chunking_route)
    app.include_router(embedding_chunks_route)
    app.include_router(build_pathrag_route)
    app.include_router(live_retrieval_route)
    app.include_router(chatbot_route)

    # Register management and monitoring routes last
    app.include_router(storage_management_route)
    app.include_router(resource_monitor_router)

    logger.info("All API routes registered successfully.")
except Exception as route_err:
    logger.critical("Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)
