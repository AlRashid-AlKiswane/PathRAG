"""
MongoDB Insert Operations Module.

This module provides functions to insert documents into MongoDB collections:
- Document chunks
- Embedding vectors
- Chatbot interaction records

Functions:
- insert_chunk_to_mongo
- insert_embed_vector_to_mongo
- insert_chatbot_entry_to_mongo

Each function logs both success and failure.

Usage:
    >>> from src.vector_db.mongo_insert_operations import *
    >>> db = get_mongo_client()["my_database"]
    >>> insert_chunk_to_mongo(db, ...)
"""

import logging
import os
import sys
import uuid
from typing import List, Optional

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.getLogger(__name__).error("Failed to set up main directory: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging

logger = setup_logging(name="MONGO-INSERT")


def insert_chunk_to_mongo(
    db,
    chunk: str,
    file: str,
    data_name: str,
    size: int,
    collection_name: str = "chunks",
    doc_id: Optional[str] = None
) -> bool:
    """
    Inserts a document chunk into MongoDB's 'chunks' collection.

    Args:
        db: MongoDB database instance.
        chunk: Text chunk.
        file: File path or name.
        data_name: Logical dataset name.
        collection_name: MongoDB collection.
        doc_id: Optional document ID.

    Returns:
        bool: True if inserted successfully.
    """
    try:
        doc_id = doc_id or str(uuid.uuid4())
        doc = {
            "_id": doc_id,
            "chunk": chunk,
            "file": file,
            "dataName": data_name,
            "size": size
        }
        db[collection_name].insert_one(doc)
        return True
    except Exception as e:
        logger.error("❌ Failed to insert chunk into MongoDB: %s", e)
        return False


def insert_embed_vector_to_mongo(
    db,
    chunk: str,
    embedding: List[float],
    chunk_id: str,
    collection_name: str = "embed_vector",
    doc_id: Optional[str] = None
) -> bool:
    """
    Inserts a vector embedding and chunk into MongoDB's 'embed_vector' collection.

    Args:
        db: MongoDB database instance.
        chunk: Original chunk text.
        embedding: List of floats.
        chunk_id: Chunk ID string.
        collection_name: MongoDB collection.
        doc_id: Optional document ID.

    Returns:
        bool: True if inserted successfully.
    """
    try:
        doc_id = doc_id or str(uuid.uuid4())
        doc = {
            "_id": doc_id,
            "chunk": chunk,
            "chunk_id": chunk_id,
            "embedding": embedding
        }
        db[collection_name].insert_one(doc)
        return True
    except Exception as e:
        logger.error("❌ Failed to insert embed vector: %s", e)
        return False


def insert_chatbot_entry_to_mongo(
    db,
    user_id: str,
    query: str,
    llm_response: str,
    retrieval_context: str,
    retrieval_rank: Optional[int],
    collection_name: str = "chatbot",
    doc_id: Optional[str] = None
) -> bool:
    """
    Inserts chatbot interaction into MongoDB's 'chatbot' collection.

    Args:
        db: MongoDB database instance.
        user_id: User ID.
        query: User query string.
        llm_response: Response from LLM.
        retrieval_context: Text chunks used in answer.
        retrieval_rank: Optional ranking.
        collection_name: MongoDB collection.
        doc_id: Optional document ID.

    Returns:
        bool: True if inserted successfully.
    """
    try:
        doc_id = doc_id or str(uuid.uuid4())
        doc = {
            "_id": doc_id,
            "user_id": user_id,
            "query": query,
            "llm_response": llm_response,
            "retrieval_context": retrieval_context,
            "retrieval_rank": retrieval_rank
        }
        db[collection_name].insert_one(doc)
        return True
    except Exception as e:
        logger.error("❌ Failed to insert chatbot entry: %s", e)
        return False


if __name__ == "__main__":
    from src.graph_db.graph_engin import get_mongo_client
    import numpy as np

    client = get_mongo_client()
    if client:
        db = client["my_project_db"]  # <-- Update to match your real DB
        logger.info("✅ MongoDB connection succeeded.")

        # Insert dummy chunk
        insert_chunk_to_mongo(
            db,
            chunk="Sample Mongo chunk",
            file="test.pdf",
            data_name="TestSet"
        )

        # Insert dummy vector
        embed = np.random.rand(384).tolist()
        insert_embed_vector_to_mongo(
            db,
            chunk="Embedding chunk",
            embedding=embed,
            chunk_id="chunk-abc-123"
        )

        # Insert dummy chatbot record
        insert_chatbot_entry_to_mongo(
            db,
            user_id="user-1",
            query="What is a vector DB?",
            llm_response="A vector DB stores high-dimensional vectors...",
            retrieval_context="Chunk text used in answer",
            retrieval_rank=1
        )

    else:
        logger.error("❌ MongoDB client not available.")
