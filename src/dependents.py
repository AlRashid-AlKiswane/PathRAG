"""
Connection Utilities

This module provides utility functions for retrieving shared application resources
from the FastAPI app state, such as the active SQLite database connection and the
LightRAG instance used for retrieval-augmented generation (RAG).

Key Features:
- Robust error handling for missing or invalid app state resources.
- Clean separation of concerns for resource access.
- Structured logging for observability and debugging.

Usage Example:
    from fastapi import Request
    from src.db.connection_utils import get_db_conn, get_light_rag, get_llm

    @app.get("/example")
    def example_endpoint(request: Request):
        conn = get_db_conn(request)
        rag = get_light_rag(request)
        llm = get_llm(request)
        # Use `conn`, `rag`, and `llm` for processing...

Raises:
- HTTPException with status 503 if the resource is not initialized.
- HTTPException with status 500 for unexpected internal errors.
"""

import logging
import os
import sys
import sqlite3

from fastapi import Request, HTTPException
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE
)

# Project path setup for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug("Main directory path configured: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

from src.infra import setup_logging
from src.rag import LightRAG
from src.llms_providers import OllamaModel

logger = setup_logging()


def get_db_conn(request: Request) -> sqlite3.Connection:
    """
    Retrieve the active SQLite database connection from FastAPI app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        sqlite3.Connection: An active database connection instance.

    Raises:
        HTTPException: 
            - 503 if the database connection is missing.
            - 500 if an unexpected error occurs.
    """
    try:
        conn = getattr(request.app.state, "conn", None)
        if not conn:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection is unavailable."
            )
        logger.debug("Database connection retrieved successfully.")
        return conn
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving database connection.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing the database connection."
        ) from e


def get_light_rag(request: Request) -> LightRAG:
    """
    Retrieve the LightRAG instance from FastAPI app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        LightRAG: An initialized LightRAG instance.

    Raises:
        HTTPException: 
            - 503 if the LightRAG instance is missing.
            - 500 if an unexpected error occurs.
    """
    try:
        rag = getattr(request.app.state, "light_rag", None)
        if not rag:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="LightRAG service is unavailable."
            )
        logger.debug("LightRAG instance retrieved successfully.")
        return rag
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving LightRAG instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing LightRAG."
        ) from e


def get_llm(request: Request) -> OllamaModel:
    """
    Retrieve the OllamaModel instance from FastAPI app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        OllamaModel: An initialized OllamaModel instance.

    Raises:
        HTTPException: 
            - 503 if the OllamaModel instance is missing.
            - 500 if an unexpected error occurs.
    """
    try:
        llm = getattr(request.app.state, "llm", None)
        if not llm:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="OllamaModel service is unavailable."
            )
        logger.debug("OllamaModel instance retrieved successfully.")
        return llm
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving OllamaModel instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing OllamaModel."
        ) from e
