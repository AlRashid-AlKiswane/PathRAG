"""
Connection Utilities

This module provides utility functions for retrieving shared application resources
from the FastAPI app state, including:

- The active SQLite database connection.
- The initialized OllamaModel used for large language model inference.
- The HuggingFaceModel instance used for embedding generation.

These utilities help centralize access to app-scoped resources while enforcing robust
error handling, observability through structured logging, and a clean separation of concerns.

Usage Example:
--------------
    from fastapi import Request
    from src.db.connection_utils import get_db_conn, get_llm, get_embedding_model

    @app.get("/example")
    def example_endpoint(request: Request):
        conn = get_db_conn(request)
        llm = get_llm(request)
        embedding_model = get_embedding_model(request)
        # Use these resources as needed...

Raises:
-------
- HTTPException with status 503 if a required resource is not initialized.
- HTTPException with status 500 if an unexpected internal error occurs.
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
from src.llms_providers import OllamaModel, HuggingFaceModel
from src.rag import PathRAG
logger = setup_logging()


def get_db_conn(request: Request) -> sqlite3.Connection:
    """
    Retrieve the active SQLite database connection from the FastAPI app state.

    Args:
        request (Request): The current FastAPI request object.

    Returns:
        sqlite3.Connection: A live SQLite database connection.

    Raises:
        HTTPException: 
            - 503 if the database connection is not available in app state.
            - 500 if an unexpected internal error occurs.
    """
    try:
        conn = getattr(request.app.state, "conn", None)
        if not conn:
            logger.error("Database connection not found in app state.")
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


def get_llm(request: Request) -> OllamaModel:
    """
    Retrieve the OllamaModel instance from FastAPI app state.

    Args:
        request (Request): The current FastAPI request object.

    Returns:
        OllamaModel: An initialized OllamaModel instance.

    Raises:
        HTTPException:
            - 503 if the OllamaModel is not available in app state.
            - 500 if an unexpected internal error occurs.
    """
    try:
        llm = getattr(request.app.state, "llm", None)
        if not llm:
            logger.error("OllamaModel instance not found in app state.")
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


def get_embedding_model(request: Request) -> HuggingFaceModel:
    """
    Retrieve the HuggingFaceModel instance from FastAPI app state.

    Args:
        request (Request): The current FastAPI request object.

    Returns:
        HuggingFaceModel: An initialized HuggingFaceModel for embeddings.

    Raises:
        HTTPException:
            - 503 if the HuggingFaceModel is not available in app state.
            - 500 if an unexpected internal error occurs.
    """
    try:
        embedding_model = getattr(request.app.state, "embedding_model", None)
        if not embedding_model:
            logger.error("HuggingFaceModel instance not found in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="HuggingFace embedding model is unavailable."
            )
        logger.debug("HuggingFaceModel instance retrieved successfully.")
        return embedding_model
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving HuggingFaceModel instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing HuggingFace embedding model."
        ) from e


def get_path_rag(request: Request) -> PathRAG:
    """
    Retrieve the PathRAG instance stored in the FastAPI application's state.

    Args:
        request (Request): The incoming FastAPI request containing application state.

    Returns:
        PathRAG: The initialized PathRAG instance.

    Raises:
        HTTPException: 
            - 503 if the PathRAG instance is not initialized or unavailable.
            - 500 for any unexpected internal server error.
    """
    try:
        path_rag = getattr(request.app.state, "path_rag", None)
        if not isinstance(path_rag, PathRAG):
            logger.error("PathRAG instance is missing or invalid in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="PathRAG service is not available. Please try again later."
            )

        logger.debug("PathRAG instance successfully retrieved from app state.")
        return path_rag

    except HTTPException:
        raise  # Already handled above

    except Exception as e:
        logger.exception("Unexpected error while retrieving PathRAG instance from app state.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error while accessing PathRAG."
        ) from e
