"""
MongoDB Collection Retrieval Module.

This module provides a function to retrieve documents from specified MongoDB collections,
supporting selective field projection and result limits.

Functions:
- pull_from_collection: Fetches data from a given MongoDB collection with optional field selection
  and document limits. Returns results as a list of dictionaries representing the documents.

Features:
- Uses PyMongo queries with projection and limit support.
- Handles PyMongo errors with full logging at various levels.
- Logs success and failure events with context.

Usage Example:
    >>> results = pull_from_collection(db, "chatbot", fields=["user_id", "query"], limit=5)
    >>> for doc in results:
    ...     print(doc["user_id"], doc["query"])
"""

import logging
import os
import sys
from typing import List, Optional, Dict, Any
from pymongo.database import Database
from pymongo.errors import PyMongoError

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to set up main directory: %s", e, exc_info=True)
    raise

from src.infra import setup_logging

logger = setup_logging(name="MONGO-SEARCH")


def pull_from_collection(
    db: Database,
    collection_name: str,
    fields: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve documents from a specific MongoDB collection with optional field projection and limit.

    Args:
        db (Database): Active MongoDB database object.
        collection_name (str): Name of the MongoDB collection.
        fields (Optional[List[str]]): List of field names to return. Defaults to all fields.
        limit (Optional[int]): Maximum number of documents to retrieve.

    Returns:
        Optional[List[Dict[str, Any]]]: 
            A list of documents (as dictionaries), or None if an error occurs.

    Example:
        >>> pull_from_collection(db, "chatbot", fields=["query", "llm_response"], limit=10)
    """
    if not isinstance(collection_name, str) or not collection_name.isidentifier():
        logger.error("Invalid collection name: %s", collection_name)
        raise ValueError(f"Invalid collection name: {collection_name}")

    try:
        logger.debug("Querying MongoDB collection '%s'", collection_name)

        collection = db[collection_name]

        projection = {field: 1 for field in fields} if fields else None
        cursor = collection.find({}, projection)

        if limit is not None:
            cursor = cursor.limit(limit)

        results = list(cursor)

        logger.info("Successfully pulled %d document(s) from '%s'", len(results), collection_name)
        return results

    except PyMongoError as e:
        logger.error("MongoDB error while querying '%s': %s", collection_name, e)
    except Exception as e:
        logger.error("Unexpected error pulling from '%s': %s", collection_name, e)
    finally:
        logger.debug("Query execution completed for collection '%s'", collection_name)

    return None


if __name__ == "__main__":
    from src.mongodb.mongodb_engin import get_mongo_client
    client = get_mongo_client()
    db = client["my_project_db"]

    results = pull_from_collection(
        db=db,
        collection_name="embed_vector",
        fields=["chunk", "chunk_id"],
        limit=10
    )

    from pprint import pprint
    pprint(results)
