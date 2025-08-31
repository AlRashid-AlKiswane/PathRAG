"""
dependents.py

This module provides dependency-injection functions for FastAPI routes to retrieve
essential services stored in the application state (`app.state`). These services
are initialized during application startup and reused across all endpoints.

Services Provided:
------------------
- get_mongo_db: Retrieves the active MongoDB client/connection.
- get_llm: Retrieves the OllamaModel instance for language generation.
- get_embedding_model: Retrieves the HuggingFaceModel instance for embeddings.
- get_path_rag: Retrieves the initialized PathRAG engine for semantic graph reasoning.

Each function checks whether the component exists in app state and raises an
appropriate HTTPException (503 if missing, 500 if error occurs during retrieval).

Usage:
------
These dependencies are designed for use with FastAPI's `Depends()` system. Example:

    from fastapi import Depends
    from .dependents import get_mongo_db

    @app.get("/documents")
    async def get_documents(db=Depends(get_mongo_db)):
        # Use db to query MongoDB
        pass

Raises:
-------
- HTTPException(503): If the component is missing in app state.
- HTTPException(500): If an unexpected internal error occurs.

Author:
-------
ALRashid AlKiswane
"""

import logging
import os
import sys
from  pymongo import MongoClient

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

def get_mongo_db(request: Request) -> MongoClient:
    """
    Retrieve the active MongoDB client/database from the FastAPI app state.

    Args:
        request (Request): The current FastAPI request object.

    Returns:
        MongoClient: The MongoDB client or database instance stored in app state.

    Raises:
        HTTPException:
            - 503 if the MongoDB client/database is not available in app state.
            - 500 if an unexpected internal error occurs.
    """
    try:
        db = getattr(request.app.state, "db", None)
        if db is None:
            logger.error("MongoDB database handle not found in app state.")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB connection is unavailable."
            )
        logger.debug("MongoDB database handle retrieved successfully.")
        return db
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving MongoDB connection.")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while accessing the MongoDB connection."
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
        # if not isinstance(path_rag, PathRAG):
        #     logger.error("PathRAG instance is missing or invalid in app state.")
        #     raise HTTPException(
        #         status_code=HTTP_503_SERVICE_UNAVAILABLE,
        #         detail="PathRAG service is not available. Please try again later."
        #     )
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
