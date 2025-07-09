"""
Main FastAPI Application Entry Point

This module initializes and configures the FastAPI application, including:
- Setting up the project base directory
- Managing application lifespan (startup/shutdown)
- Initializing SQLite database connection and tables
- Instantiating core components such as OllamaModel and HuggingFaceModel
- Mounting all API routes

Features:
- Robust error handling with critical logging on failure
- Async lifespan context for clean resource management
- Structured logger for full observability
"""

import asyncio
import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug(f"📁 Project base path set to: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical(
        "🔴 [Startup Critical] Failed to set up project path. Error: %s\nSystem paths: %s",
        e, sys.path,
        exc_info=True
    )
    sys.exit(1)

# === Internal Imports (after path config) ===
from src.db import (
    get_sqlite_engine,
    init_chunks_table,
    init_embed_vector_table,
    init_entitiys_table,
)
from src.routes import (
    upload_route,
    chunking_route,
    embedding_chunks_route,
    ner_route,
    live_retrieval_route
)
from src.infra import setup_logging
from src.llms_providers import OllamaModel, HuggingFaceModel, NERModel
from src.rag import FaissRAG, EntityLevelFiltering
from src.helpers import get_settings, Settings

# === Logger and Settings ===
logger = setup_logging()
app_settings: Settings = get_settings()

# === FastAPI App ===
app = FastAPI(title="Graph-RAG API", version="1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI app lifecycle events.

    On startup:
    - Establish database connection
    - Initialize tables
    - Load LLM and embedding model

    On shutdown:
    - Close the database connection safely

    Raises:
        SystemExit: If the database connection cannot be established.
    """
    try:
        logger.info("🚀 [Startup] Initializing FastAPI application...")

        # Establish SQLite DB connection
        conn = get_sqlite_engine()
        if not conn:
            logger.critical("❌ [Startup Error] Unable to establish database connection.")
            sys.exit(1)

        app.state.conn = conn
        logger.debug("🔗 Database connection established.")

        # Initialize database tables
        init_chunks_table(conn=conn)
        logger.info("🧱 'chunks' table initialized.")
        init_embed_vector_table(conn=conn)
        logger.info("🧠 'embed_vector' table initialized.")
        init_entitiys_table(conn=conn)
        logger.info(" 'enttesy' table initialized.")

        # Load LLM and embedding models
        app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
        logger.info(f"🤖 OllamaModel initialized: {app_settings.OLLAMA_MODEL}")
        app.state.embedding_model = HuggingFaceModel(model_name=app_settings.EMBEDDING_MODEL)
        logger.info(f"🔡 HuggingFace embedding model loaded: {app_settings.EMBEDDING_MODEL}")
        ner_model = NERModel(model_name=app_settings.NER_MODEL)
        app.state.ner_model = ner_model

        logger.info("Successfully Loading NER Model: %s", app_settings.NER_MODEL)
        app.state.faiss_rag = FaissRAG(conn=conn)

        app.state.entity_level_filtering = EntityLevelFiltering(conn=conn,
                                                                ner_model=ner_model)
        yield  # App is now ready to serve

    except Exception as e:
        logger.exception("💥 [Startup Exception] Critical error during startup.")
        raise

    finally:
        # Graceful shutdown
        conn = getattr(app.state, "conn", None)
        if conn:
            try:
                conn.close()
                logger.info("🛑 [Shutdown] Database connection closed.")
            except Exception as e:
                logger.error("⚠️ [Shutdown Error] Failed to close database connection: %s", e)


# === Register App Lifespan ===
app.router.lifespan_context = lifespan

# === Register API Routes ===
app.include_router(upload_route)
app.include_router(chunking_route)
app.include_router(embedding_chunks_route)
app.include_router(ner_route)
app.include_router(live_retrieval_route)