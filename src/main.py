"""
Main FastAPI Application Entry Point

This module initializes and configures the FastAPI application, including:
- Setting up the project base directory
- Managing application lifespan (startup/shutdown)
- Initializing SQLite database connection and tables
- Instantiating core components such as OllamaModel and LightRAG
- Mounting API routes

Features:
- Robust error handling with critical logging on failure
- Async lifespan context management for graceful startup/shutdown
- Structured logger integration for observability
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

# Setup project base path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug(f"Project base path set to: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical(
        "[Startup Critical] Failed to set up project base path. "
        f"Error: {e}. System paths: {sys.path}",
        exc_info=True
    )
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.db import get_sqlite_engine, init_chunks_table
from src.routes import (
    upload_route,
    chunking_route,
    chunks_to_rag_route,
    live_retrieval_route,
)
from src.infra import setup_logging
from src.rag import LightRAG
from src.llms_providers import OllamaModel
from src.helpers import get_settings, Settings

logger = setup_logging()
app_settings: Settings = get_settings()

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI app lifespan.

    On startup:
    - Establish a SQLite database connection and assign it to app state.
    - Initialize the required database tables.
    - Instantiate OllamaModel and LightRAG, storing them in app state.

    On shutdown:
    - Close the database connection if it exists.

    Raises:
        SystemExit: If the database connection fails to initialize.
        Exception: For unexpected errors during startup.
    """
    try:
        logger.info("[Startup] Initializing application...")
        conn = get_sqlite_engine()
        if not conn:
            logger.critical("[Startup Error] Unable to establish database connection.")
            sys.exit(1)
        app.state.conn = conn
        logger.debug("[Startup] Database connection established.")

        init_chunks_table(conn=conn)
        logger.info("[Startup] Database chunks table initialized.")

        app.state.llm = OllamaModel(app_settings.OLLAMA_MODEL)
        logger.info(f"[Startup] OllamaModel initialized with model: {app_settings.OLLAMA_MODEL}")

        app.state.light_rag = LightRAG(
            embedding_model_name=app_settings.EMBEDDING_MODEL,
            llm=app.state.llm
        )
        logger.info(f"[Startup] LightRAG initialized with embedding model: {app_settings.EMBEDDING_MODEL}")

        yield  # Allow FastAPI to start serving requests

    except Exception as e:
        logger.exception(f"[Startup Exception] Failed during application startup: {e}")
        raise

    finally:
        conn = getattr(app.state, "conn", None)
        if conn:
            try:
                conn.close()
                logger.info("[Shutdown] Database connection closed successfully.")
            except Exception as e:
                logger.error(f"[Shutdown Error] Error closing database connection: {e}")


# Assign the lifespan context manager to FastAPI app
app.router.lifespan_context = lifespan

# Mount API routes
app.include_router(upload_route)
app.include_router(chunking_route)
app.include_router(chunks_to_rag_route)
app.include_router(live_retrieval_route)
