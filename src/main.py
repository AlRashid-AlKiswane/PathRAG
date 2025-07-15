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

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug(f"📁 Project base path configured: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical("🔴 Failed to configure project path: %s\nSystem path: %s", e, sys.path, exc_info=True)
    sys.exit(1)

# === Internal Imports (after path configuration) ===
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
        resource_monitor_router
    )
    from src.infra import setup_logging
    from src.llms_providers import OllamaModel, HuggingFaceModel, NERModel
    from src.rag import FaissRAG, EntityLevelFiltering
    from src.helpers import get_settings, Settings
except ImportError as e:
    logging.critical("🔴 Failed to import internal modules: %s", e, exc_info=True)
    sys.exit(1)

# === Logger and Settings Initialization ===
logger = setup_logging()
app_settings: Settings = get_settings()

# === Lifespan Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Starting up Graph-RAG API...")

        # Connect to SQLite DB
        conn = get_sqlite_engine()
        if not conn:
            raise ConnectionError("Failed to get a valid SQLite connection.")
        app.state.conn = conn
        logger.info("🔗 SQLite database connection established.")

        # Initialize tables
        init_chunks_table(conn)
        init_embed_vector_table(conn)
        init_entities_table(conn)
        init_chatbot_table(conn)
        logger.info("✅ All database tables initialized successfully.")

        # Load models
        app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
        app.state.embedding_model = HuggingFaceModel(app_settings.EMBEDDING_MODEL)
        ner_model = NERModel(model_name=app_settings.NER_MODEL)
        app.state.ner_model = ner_model
        logger.info("✅ All models loaded successfully.")

        # Initialize FAISS RAG
        try:
            faiss_rag = FaissRAG(conn)
            await faiss_rag.initialize_faiss()
            app.state.faiss_rag = faiss_rag
            logger.info("🔍 FAISS RAG initialized and ready.")
        except Exception as rag_err:
            logger.critical("❌ Failed to initialize FAISS RAG: %s", rag_err, exc_info=True)
            raise

        # Entity-level filtering
        app.state.entity_level_filtering = EntityLevelFiltering(conn=conn, ner_model=ner_model)

        yield  # App runs here

    except Exception:
        logger.exception("💥 Fatal error during FastAPI startup.")
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
    description="A lightweight RAG system with semantic and entity-level filtering",
    lifespan=lifespan  # ✅ use modern lifespan pattern
)

# === Register API Routers ===
try:
    app.include_router(upload_route)
    logger.debug("🔗 Registered upload route.")

    app.include_router(chunking_route)
    logger.debug("✂️ Registered chunking route.")

    app.include_router(embedding_chunks_route)
    logger.debug("📎 Registered embedding route.")

    app.include_router(ner_route)
    logger.debug("🔍 Registered NER route.")

    app.include_router(live_retrieval_route)
    logger.debug("⚡ Registered live retrieval route.")

    app.include_router(storage_management_route)
    logger.debug("🗂️ Registered Storage Management route.")

    app.include_router(chatbot_route)
    logger.debug("🤖 Registered Chatbot route.")

    app.include_router(resource_monitor_router)
    logger.debug("Registered Resource Monitor route.")

    logger.info("✅ All API routes registered.")
except Exception as route_err:
    logger.critical("❌ Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)
