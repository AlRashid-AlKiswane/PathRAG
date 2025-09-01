"""
MongoDB Collection Initialization Module.

This module defines and initializes MongoDB collections that mirror traditional
Qdrant collections: chunks, embed_vector, and chatbot.

Features:
- Checks for existence and creates MongoDB collections if missing.
- Optionally sets indexes or schema validation (basic).
- Logs success and failure for all setup actions.

Usage Example:
    >>> from src.vector_db.mongo_init_collections import (
    ...     init_chunks_collection,
    ...     init_embed_vector_collection,
    ...     init_chatbot_collection
    ... )
    >>> client = get_mongo_client()
    >>> db = client["your_database_name"]
    >>> init_chunks_collection(db)
"""

import logging
import os
import sys
from pymongo import MongoClient, errors

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.getLogger(__name__).error("Failed to set up main directory: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.helpers import get_settings, Settings

logger = setup_logging(name="MONGO-COLLECTIONS")
app_settings: Settings = get_settings()


def init_chunks_collection(db) -> None:
    """
    Initializes the 'chunks' collection in MongoDB.

    Example Schema:
    - chunk: str
    - file: str
    - dataName: str
    - size: int
    """
    try:
        if "chunks" not in db.list_collection_names():
            db.create_collection("chunks")
            logger.info("'chunks' collection created.")
        else:
            logger.info("'chunks' collection already exists.")
    except Exception as e:
        logger.exception("Failed to initialize 'chunks' collection: %s", e)
        raise


def init_embed_vector_collection(db) -> None:
    """
    Initializes the 'embed_vector' collection in MongoDB.

    Example Schema:
    - chunk: str
    - chunk_id: str
    - embedding: list[float]
    """
    try:
        if "embed_vector" not in db.list_collection_names():
            db.create_collection("embed_vector")
            logger.info("'embed_vector' collection created.")
        else:
            logger.info("'embed_vector' collection already exists.")
    except Exception as e:
        logger.exception("Failed to initialize 'embed_vector' collection: %s", e)
        raise


def init_chatbot_collection(db) -> None:
    """
    Initializes the 'chatbot' collection in MongoDB.

    Example Schema:
    - user_id: str
    - query: str
    - llm_response: str
    - retrieval_context: str
    - retrieval_rank: int
    - created_at: str
    """
    try:
        if "chatbot" not in db.list_collection_names():
            db.create_collection("chatbot")
            logger.info("'chatbot' collection created.")
        else:
            logger.info("'chatbot' collection already exists.")
    except Exception as e:
        logger.exception("Failed to initialize 'chatbot' collection: %s", e)
        raise


if __name__ == "__main__":
    from src.mongodb.mongodb_engin import get_mongo_client  # your mongo_engine.py

    client = get_mongo_client()
    if client:
        try:
            db = client["pathragdb"]
            logger.info(f"Initializing MongoDB collections in DB: {db.name} ...")
            init_chunks_collection(db)
            init_embed_vector_collection(db)
            init_chatbot_collection(db)
            logger.info("All collections initialized.")
        except Exception as e:
            logger.error("Failed to initialize collections: %s", e)
            sys.exit(1)
    else:
        logger.error("Could not connect to MongoDB.")
        sys.exit(1)
