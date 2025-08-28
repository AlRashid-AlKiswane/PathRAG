"""
Module for safely clearing MongoDB collections.

This module provides a utility function to remove all documents from a specified
MongoDB collection with proper name validation, error handling, and structured logging.

Usage:
    >>> from pymongo import MongoClient
    >>> from src.infra.db_clear_mongo import clear_collection
    >>> client = MongoClient("mongodb://localhost:27017")
    >>> db = client["mydatabase"]
    >>> clear_collection(db, "users")
"""

import os
import sys
import logging
from pymongo.database import Database
from pymongo.errors import PyMongoError

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings

# Initialize application settings and logger
logger = setup_logging(name="MONGO-CLEAR")
app_settings: Settings = get_settings()


def clear_collection(db: Database, collection_name: str) -> None:
    """
    Delete all documents from a specified MongoDB collection.

    This function validates the collection name to ensure it is a proper identifier.
    It then deletes all documents from the collection and logs the result.

    Args:
        db (Database): The active MongoDB database object.
        collection_name (str): The name of the collection to clear.

    Raises:
        ValueError: If the collection name is invalid.
        RuntimeError: If a MongoDB error occurs during deletion.

    Side Effects:
        Logs success or failure.
    """
    if not isinstance(collection_name, str) or not collection_name.isidentifier():
        error_msg = f"Invalid collection name: {collection_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.debug("Starting to clear collection '%s'", collection_name)
        result = db[collection_name].delete_many({})
        logger.info("Successfully cleared collection '%s' (deleted %d documents)",
                    collection_name, result.deleted_count)

    except PyMongoError as e:
        error_msg = f"MongoDB error clearing collection '{collection_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    except Exception as e:
        error_msg = f"Unexpected error clearing collection '{collection_name}': {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    from src.mongodb.mongodb_engin import get_mongo_client

    client = get_mongo_client()
    db = client["my_project_db"]

    clear_collection(db, "embed_vector")
    clear_collection(db, "chunks")
    clear_collection(db, "chatbot")
