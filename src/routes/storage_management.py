"""
storage_management_route.py

This module defines an API route for managing storage tables in the system's MongoDB database.
It provides options to clear specific collections or perform a full reset of all relevant data collections.

Endpoint:
    POST /api/v1/management-storage/

Parameters:
    - do_erase_all (bool): Clears all relevant collections: `chunks`, `embed_vector`, and `chatbot`.
    - reset_chunks (bool): Clears only the `chunks` collection (default: True).
    - reset_embeddings (bool): Clears only the `embed_vector` collection.
    - reset_chatbot_history (bool): Clears only the `chatbot` collection.

Returns:
    JSONResponse: A message indicating which collections were successfully reset.
    - 200: Collections were cleared successfully.
    - 400: No reset flags were provided; no action taken.
    - 500: A database or unexpected server error occurred.

Functionality:
    - Uses dependency-injected MongoDB client.
    - Calls `clear_collection()` utility from `src.graph_db` to delete collection contents.
    - Logs each operation at info/debug/warning levels.
    - Handles database and operational errors gracefully.

Dependencies:
    - FastAPI
    - MongoDB (via pymongo)
    - Custom project utilities (`clear_collection`, `setup_logging`, `get_mongo_db`)
"""

import os
import sys
import logging
from typing import List

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.mongodb import clear_collection
from src.infra import setup_logging
from src import get_mongo_db

# === Logger Setup ===
logger = setup_logging(name="MANAGEMENT-STORAGE")

# === API Router ===
storage_management_route = APIRouter(
    prefix="/api/v1/management-storage",
    tags=["Management Storage"],
    responses={404: {"description": "Not found"}},
)


@storage_management_route.post("", response_class=JSONResponse)
async def storage_management(
    do_erase_all: bool = False,
    reset_chunks: bool = True,
    reset_embeddings: bool = False,
    reset_chatbot_history: bool = False,
    db: MongoClient = Depends(get_mongo_db),
) -> JSONResponse:
    """
    Manage and optionally reset storage collections in the MongoDB database.

    Args:
        do_erase_all (bool): If True, clears all relevant collections: 'chunks', 'embed_vector', 'chatbot'.
        reset_chunks (bool): If True, clears only the 'chunks' collection.
        reset_embeddings (bool): If True, clears only the 'embed_vector' collection.
        reset_chatbot_history (bool): If True, clears only the 'chatbot' collection.
        db (MongoClient): MongoDB client, injected by dependency.

    Returns:
        JSONResponse: Operation result with a list of affected collections and success message.

    Raises:
        HTTPException: With appropriate status code for database or unexpected errors.
    """
    try:
        affected_collections: List[str] = []

        if do_erase_all:
            collections = ["chunks", "embed_vector", "chatbot"]
            logger.info("Initiating full reset of all storage collections.")
            for collection in collections:
                clear_collection(db=db, collection_name=collection)
                affected_collections.append(collection)
                logger.debug("Cleared collection: %s", collection)
        else:
            if reset_chunks:
                clear_collection(db=db, collection_name="chunks")
                affected_collections.append("chunks")
                logger.debug("Cleared collection: chunks")

            if reset_embeddings:
                clear_collection(db=db, collection_name="embed_vector")
                affected_collections.append("embed_vector")
                logger.debug("Cleared collection: embed_vector")

            if reset_chatbot_history:
                clear_collection(db=db, collection_name="chatbot")
                affected_collections.append("chatbot")
                logger.debug("Cleared collection: chatbot")

        if not affected_collections:
            logger.warning("No collection reset flags set. No collections were cleared.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "No reset flag provided. Specify at least one collection to reset."},
            )

        logger.info("Storage management operation completed. Collections affected: %s", affected_collections)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Collections reset successfully.",
                "collections_affected": affected_collections,
            },
        )

    except PyMongoError as mongo_err:
        logger.error("Database error during collection reset: %s", mongo_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred during collection reset.",
        )

    except Exception as e:
        logger.critical("Unexpected error in storage management: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during storage management.",
        )
