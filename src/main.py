"""
Main FastAPI Application Entry Point

This module initializes the FastAPI application, configures routes, sets up logging,
and manages the application lifecycle including database initialization.

Features:
- Sets up project base directory
- Initializes SQLite connection on startup
- Initializes required database tables
- Mounts upload routes
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Setup project base path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    sys.path.append(MAIN_DIR)
    logging.debug(f"Project base path set to: {MAIN_DIR}")
except (ImportError, OSError) as e:
    logging.critical(
        "[Startup Critical] Failed to set up project base path. "
        f"Error: {str(e)}. System paths: {sys.path}",
        exc_info=True
    )
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.db import get_sqlite_engine, init_chunks_table
from src.routes import (upload_route,
                        chunking_route)
from src.infra import setup_logging

# Set up application logger
logger = setup_logging()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lambda app: lifespan(app))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    On startup:
    - Establishes a SQLite connection
    - Initializes required database tables

    On shutdown:
    - Closes the database connection gracefully
    """
    try:
        logger.info("[Startup] Application starting...")
        conn = get_sqlite_engine()
        if not conn:
            logger.critical("[Startup Error] Failed to establish database connection.")
            sys.exit(1)
        app.state.conn = conn
        logger.debug("[Startup] Database connection established.")

        init_chunks_table(conn=conn)
        logger.info("[Startup] Chunks table initialized successfully.")

        yield

    except Exception as e:
        logger.exception(f"[Startup Exception] {e}")
        raise
    finally:
        if hasattr(app.state, "conn") and app.state.conn:
            app.state.conn.close()
            logger.info("[Shutdown] Database connection closed.")


# Mount API routes
app.include_router(upload_route)
app.include_router(chunking_route)
