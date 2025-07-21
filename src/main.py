"""
Graph-RAG API Entry Point

This module sets up and runs the FastAPI application for the Graph-RAG system.

Key responsibilities:
- Configure project base directory for imports
- Initialize structured logging
- Manage application lifecycle (startup/shutdown)
- Connect to and initialize the SQLite database
- Load and initialize core NLP models
- Mount route modules for RAG processing

Author: [Your Name]
Created: [YYYY-MM-DD]
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
    logging.debug(f"📁 Project base path configured: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical("🔴 Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Internal Imports ===
try:
    from src.db import (
        get_sqlite_engine,
        init_chunks_table,
        init_embed_vector_table,
        init_entities_table,
        init_chatbot_table
    )
    from src.routes import (
        upload_route,
        chunking_route,
        embedding_chunks_route,
        ner_route,
        live_retrieval_route,
        storage_management_route,
        chatbot_route,
        resource_monitor_router,
        build_pathrag_route
    )
    from src.infra import setup_logging
    from src.llms_providers import OllamaModel, HuggingFaceModel
    from src.rag import PathRAG
    from src.helpers import get_settings, Settings
except ImportError as e:
    logging.critical("🔴 Failed to import internal modules: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging()
app_settings: Settings = get_settings()

# === Lifespan Event Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Starting up Graph-RAG API...")

        # Database connection
        conn = get_sqlite_engine()
        if not conn:
            raise ConnectionError("❌ Failed to get a valid SQLite connection.")
        app.state.conn = conn
        logger.info("🔗 SQLite database connection established.")

        # Database table initialization
        init_chunks_table(conn)
        init_embed_vector_table(conn)
        init_entities_table(conn)
        init_chatbot_table(conn)
        logger.info("✅ All database tables initialized successfully.")

        # Load LLM and embedding model
        app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
        app.state.embedding_model = HuggingFaceModel(app_settings.EMBEDDING_MODEL)
        logger.info("✅ All models loaded successfully.")

        # Initialize RAG
        try:
            path_rag = PathRAG(embedding_model=app.state.embedding_model,
                               decay_rate=app_settings.DECAY_RATE,
                               prune_thresh=app_settings.PRUNE_THRESH,
                               sim_should=app_settings.SIM_THRESHOLD)

            app.state.path_rag = path_rag
            logger.info("🔍 PathRAG initialized and ready.")
        except Exception as rag_err:
            logger.critical("❌ Failed to initialize PathRAG: %s", rag_err, exc_info=True)
            raise

        yield

    except Exception as e:
        logger.exception("💥 Fatal error during FastAPI startup: %s", e)
        raise

    finally:
        # Graceful shutdown
        conn = getattr(app.state, "conn", None)
        if conn:
            try:
                conn.close()
                logger.info("🛑 SQLite connection closed.")
            except Exception as e:
                logger.warning("⚠️ Error closing DB connection: %s", e, exc_info=True)

# === FastAPI Application ===
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="A Path-aware Retrieval-Augmented Generation system using semantic graphs.",
    lifespan=lifespan
)

# === Register Routes ===
try:
    app.include_router(upload_route)
    app.include_router(chunking_route)
    app.include_router(embedding_chunks_route)
    app.include_router(ner_route)
    app.include_router(live_retrieval_route)
    app.include_router(storage_management_route)
    app.include_router(chatbot_route)
    app.include_router(resource_monitor_router)
    app.include_router(build_pathrag_route)
    logger.info("✅ All API routes registered successfully.")
except Exception as route_err:
    logger.critical("❌ Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)
