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
from src.llms_providers import OllamaModel, HuggingFaceModel, NERModel
from src.rag import FaissRAG, EntityLevelFiltering
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


def get_ner_model(request: Request) -> NERModel:
    """
    Dependency function to retrieve the NERModel instance from FastAPI app state.

    This function is intended to be used with FastAPI's dependency injection system,
    allowing route handlers to access the NER model without re-instantiating it.

    Args:
        request (Request): The FastAPI request object, containing app state.

    Returns:
        NERModel: An instance of the pre-loaded NERModel from app.state.

    Raises:
        HTTPException: 
            - 503 if the NER model is not available in app state.
            - 500 if an unexpected error occurs during retrieval.
    """
    try:
        ner_model = getattr(request.app.state, "ner_model", None)
        if not ner_model:
            logger.error("NERModel instance not found in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="NER model is not available. Try again later."
            )
        logger.debug("NERModel instance retrieved successfully from app state.")
        return ner_model

    except HTTPException:
        raise  # Already logged and constructed correctly above

    except Exception as e:
        logger.exception("Unexpected error while retrieving NERModel instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error while accessing the NER model."
        ) from e


def get_faiss_rag(request: Request) -> FaissRAG:
    """
    Dependency function to retrieve the FaissRAG instance from FastAPI app state.

    This function allows FastAPI route handlers to access the shared FaissRAG instance
    stored in the application's state without re-instantiating it.

    Args:
        request (Request): The FastAPI request object, which contains app state.

    Returns:
        FaissRAG: An instance of the pre-loaded FaissRAG from app.state.

    Raises:
        HTTPException:
            - 503 if the FaissRAG instance is not available in app state.
            - 500 if an unexpected error occurs during retrieval.
    """
    try:
        faiss_rag = getattr(request.app.state, "faiss_rag", None)
        if not faiss_rag:
            logger.error("FaissRAG instance not found in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="FaissRAG is not available. Try again later."
            )

        logger.debug("FaissRAG instance retrieved successfully from app state.")
        return faiss_rag

    except HTTPException:
        raise  # Already handled and logged above

    except Exception as e:
        logger.exception("Unexpected error while retrieving FaissRAG instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error while accessing the FaissRAG instance."
        ) from e


def get_entity_level_filtering(request: Request) -> EntityLevelFiltering:
    """
    Dependency function to retrieve the EntityLevelFiltering instance from FastAPI app state.

    This function allows FastAPI route handlers to access the shared EntityLevelFiltering instance
    stored in the application's state without re-instantiating it.

    Args:
        request (Request): The FastAPI request object, which contains app state.

    Returns:
        EntityLevelFiltering: An instance of the pre-loaded EntityLevelFiltering from app.state.

    Raises:
        HTTPException:
            - 503 if the EntityLevelFiltering instance is not available in app state.
            - 500 if an unexpected error occurs during retrieval.
    """
    try:
        entity_filtering = getattr(request.app.state, "entity_level_filtering", None)
        if not entity_filtering:
            logger.error("EntityLevelFiltering instance not found in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="EntityLevelFiltering is not available. Try again later."
            )

        logger.debug("EntityLevelFiltering instance retrieved successfully from app state.")
        return entity_filtering

    except HTTPException:
        raise  # Already handled and logged above

    except Exception as e:
        logger.exception("Unexpected error while retrieving EntityLevelFiltering instance.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error while accessing the EntityLevelFiltering instance."
        ) from e
