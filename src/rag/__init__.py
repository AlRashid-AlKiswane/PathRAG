"""
rag package

This package implements Retrieval-Augmented Generation (RAG) functionalities,
including graph-based reasoning with the PathRAG module.

Modules:
- pathrag: Contains the PathRAG class for semantic graph construction,
  relational path retrieval, and decay-based path scoring for prompt generation.

Imports:
- PathRAG: Primary class implementing the Path-aware Retrieval-Augmented Generation logic.
"""

from .pathrag import PathRAG

__all__ = ["PathRAG"]
