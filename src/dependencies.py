"""
This module provides utility functions to retrieve core application components
from the FastAPI application state during request handling. 

Specifically, it handles access to:

- SQLite database connection (`sqlite3.Connection`):  
  Ensures a persistent connection is available for executing database queries.

- Ollama language model instance (`OllamaModel`):  
  Provides access to the configured LLM backend for natural language generation.

- HuggingFace embedding model instance (`HuggingFaceModel`):  
  Used for embedding generation for semantic search or vector-based retrieval.

- PathRAG instance (`PathRAG`):  
  Supports Path-aware Retrieval-Augmented Generation for complex multi-hop reasoning.

Each retrieval function accepts the current FastAPI `Request` object, extracts the 
corresponding component from the `app.state` container, performs validation, and 
raises an HTTPException with appropriate status codes and error messages in case 
of missing or invalid components.

Error Handling:
- If a requested component is not found or invalid in the application state, 
  a 503 Service Unavailable response is raised.
- If an unexpected error occurs during retrieval, a 500 Internal Server Error 
  is raised with logging of the exception details.

Logging:
- All retrieval attempts, successes, and failures are logged using the configured 
  logger.
- Critical failures during project path setup cause the application to exit 
  with logged stack traces.

Project Path Setup:
- Adjusts the Python module search path to include the main project directory, 
  enabling relative imports from the `src` package.

Dependencies:
- `fastapi.Request` and `fastapi.HTTPException` for request context and error handling.
- `sqlite3` for database connection typing.
- Project-specific modules:
  - `src.infra.setup_logging` for logger configuration.
  - `src.llms_providers.OllamaModel` and `HuggingFaceModel` for LLM and embedding models.
  - `src.rag.PathRAG` for the Path-aware RAG system.

Usage:
These functions are intended to be used inside FastAPI route handlers or middleware
to safely obtain core service instances from the application state, abstracting away
direct state access and centralizing error handling.

Example:
    def some_route_handler(request: Request):
        db_conn = get_db_conn(request)
        llm = get_llm(request)
        embedding_model = get_embedding_model(request)
        path_rag = get_path_rag(request)
        # use these components to process the request...

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
logger = setup_logging(name="DEPENDENTS")

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
