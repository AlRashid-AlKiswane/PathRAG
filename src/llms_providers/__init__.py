"""
llms_providers

This package provides interfaces to local and cloud-based large language models (LLMs)
and embedding models used within the PathRAG-LightRAG system.

Modules:
- ollama_provider: Interface for managing and querying local Ollama LLMs.
- embedding: Embedding generator using HuggingFace SentenceTransformer.

Exports:
- OllamaModel: High-level client for interacting with the Ollama LLM service.
- HuggingFaceModel: Wrapper for generating text embeddings using pretrained models.
"""

from .ollama_provider import OllamaModel
from .embedding import HuggingFaceModel

__all__ = [
    "OllamaModel",
    "HuggingFaceModel",
]
