"""
Graph Database Integration Package.

This module provides initialization, insertion, and clearing utilities for working with
MongoDB collections in a graph-aware Retrieval-Augmented Generation (RAG) pipeline.

Submodules:
- `graph_engin`: Initializes the MongoDB client connection.
- `graph_collection`: Creates and configures required MongoDB collections:
    - chunks
    - embed_vector
    - chatbot
- `graph_insert`: Provides insert functions for:
    - Document chunks
    - Embedding vectors
    - Chatbot interaction logs
- `graph_clear_collection`: Safely clears documents from any MongoDB collection.

Typical Workflow:
1. Use `get_mongo_client()` to connect to your MongoDB instance.
2. Initialize collections with `init_*_collection()` functions.
3. Insert documents using the `insert_*_to_mongo()` functions.
4. Clear collections when needed with `clear_collection()`.

Example:
    >>> from src.graph_db import get_mongo_client, insert_chunk_to_mongo
    >>> db = get_mongo_client()
    >>> insert_chunk_to_mongo(db, "Some chunk", "file.pdf", "MyCorpus")
"""

from .graph_clear_collection import clear_collection
from .graph_collection import (
    init_chatbot_collection,
    init_chunks_collection,
    init_embed_vector_collection,
)
from .graph_insert import (
    insert_chatbot_entry_to_mongo,
    insert_chunk_to_mongo,
    insert_embed_vector_to_mongo,
)
from .graph_engin import get_mongo_client

__all__ = [
    "clear_collection",
    "init_chatbot_collection",
    "init_chunks_collection",
    "init_embed_vector_collection",
    "insert_chatbot_entry_to_mongo",
    "insert_chunk_to_mongo",
    "insert_embed_vector_to_mongo",
    "get_mongo_client",
]
