"""
schema package
==============

This package defines data models, configuration classes, and processing 
schemas used across the PathRAG system. It provides structured request/response 
objects, configuration models, and OCR/chunking schemas to ensure consistency 
across the pipeline.

The exports cover:
- Chatbot interface for user interaction.
- OCR engine and results for text extraction.
- Chunking requests/responses and processing configuration.
- PathRAG configuration for retrieval-augmented generation pipelines.
"""

from .chatbot import Chatbot
from .md_chunks import ChunkRequest, ChunkResponse
from .rag import PathRAGConfig
from .ocr_core import OCREngine, OCRResult
from .chunker_route import (
    ChunkingRequest,
    ChunkingResponse,
    ProcessingConfig,
    ProcessingMode,
)

__all__ = [
    # Chatbot
    "Chatbot",

    # Chunking
    "ChunkRequest",
    "ChunkResponse",
    "ChunkingRequest",
    "ChunkingResponse",
    "ProcessingConfig",
    "ProcessingMode",

    # RAG
    "PathRAGConfig",

    # OCR
    "OCREngine",
    "OCRResult",
]
