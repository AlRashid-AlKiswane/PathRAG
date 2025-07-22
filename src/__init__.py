"""
src package

This package contains the core modules and components of the Graph-RAG system,
including database utilities, language model providers, routing definitions,
retrieval-augmented generation logic, and helper functions.

The `dependents` module provides accessor functions to retrieve
singleton service instances from the FastAPI application state,
such as the database connection, language models, embedding models,
and the PathRAG instance.

Imported here for convenient direct access:

- get_db_conn: Retrieve the SQLite connection from FastAPI state.
- get_llm: Retrieve the Ollama LLM instance from FastAPI state.
- get_embedding_model: Retrieve the HuggingFace embedding model.
- get_path_rag: Retrieve the PathRAG instance.

"""

from .dependencies import (
    get_mongo_db,
    get_llm,
    get_embedding_model,
    get_path_rag
)

__all__ = [
    "get_db_conn",
    "get_llm",
    "get_embedding_model",
    "get_path_rag"
]