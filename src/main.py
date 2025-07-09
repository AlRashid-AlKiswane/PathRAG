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

import asyncio
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
    logging.critical(
        "🔴 Failed to configure project path: %s\nSystem path: %s", e, sys.path, exc_info=True
    )
    sys.exit(1)

# === Internal Imports (after path configuration) ===
try:
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
        live_retrieval_route,
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

# === FastAPI Application ===
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="A lightweight RAG system with semantic and entity-level filtering"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager handling the FastAPI application lifespan.

    This includes:
    - Startup phase: DB connection, table creation, model loading
    - Shutdown phase: DB disconnection

    Args:
        app (FastAPI): The FastAPI application instance

    Yields:
        None

    Raises:
        SystemExit: If database setup or critical components fail
    """
    try:
        logger.info("🚀 Starting up Graph-RAG API...")

        # Connect to SQLite DB
        try:
            conn = get_sqlite_engine()
            if not conn:
                raise ConnectionError("Failed to get a valid SQLite connection.")
            app.state.conn = conn
            logger.info("🔗 SQLite database connection established.")
        except Exception as db_err:
            logger.critical("❌ Could not establish database connection: %s", db_err, exc_info=True)
            sys.exit(1)

        # Initialize tables
        try:
            init_chunks_table(conn)
            logger.debug("📦 Initialized 'chunks' table.")

            init_embed_vector_table(conn)
            logger.debug("📊 Initialized 'embed_vector' table.")

            init_entitiys_table(conn)
            logger.debug("🧠 Initialized 'entitsy' table.")

            logger.info("✅ All database tables initialized successfully.")
        except Exception as table_err:
            logger.error("❌ Failed to initialize tables: %s", table_err, exc_info=True)
            sys.exit(1)

        # Load models
        try:
            app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
            logger.info(f"🤖 OllamaModel loaded: {app_settings.OLLAMA_MODEL}")

            app.state.embedding_model = HuggingFaceModel(model_name=app_settings.EMBEDDING_MODEL)
            logger.info(f"🔡 Embedding model loaded: {app_settings.EMBEDDING_MODEL}")

            ner_model = NERModel(model_name=app_settings.NER_MODEL)
            app.state.ner_model = ner_model
            logger.info(f"🧬 NER model loaded: {app_settings.NER_MODEL}")
        except Exception as model_err:
            logger.critical("❌ Failed to load one or more models: %s", model_err, exc_info=True)
            sys.exit(1)

        # Initialize RAG components
        try:
            app.state.faiss_rag = FaissRAG(conn=conn)
            logger.debug("🔍 FAISS RAG component initialized.")

            app.state.entity_level_filtering = EntityLevelFiltering(
                conn=conn,
                ner_model=ner_model
            )
            logger.debug("📎 Entity-level filtering module initialized.")
        except Exception as rag_err:
            logger.error("❌ Failed to initialize RAG components: %s", rag_err, exc_info=True)
            sys.exit(1)

        logger.info("✅ FastAPI app fully initialized and ready.")
        yield

    except Exception as e:
        logger.exception("💥 Fatal error during FastAPI startup.")
        raise

    finally:
        # Clean shutdown
        conn = getattr(app.state, "conn", None)
        if conn:
            try:
                conn.close()
                logger.info("🛑 SQLite database connection closed gracefully.")
            except Exception as close_err:
                logger.warning("⚠️ Error closing DB connection: %s", close_err, exc_info=True)


# === Register App Lifespan Context ===
app.router.lifespan_context = lifespan

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

    logger.info("✅ All API routes registered.")
except Exception as route_err:
    logger.critical("❌ Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)
