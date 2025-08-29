"""
controllers/__init__.py

This module aggregates and exposes the core controller functions for document
processing and file management.

Exports:
- chunking_docs: Function to load and split documents into manageable chunks.
- generate_unique_filename: Utility to generate a sanitized, unique filename 
  based on the original filename, including timestamp and UUID suffix.

This allows users to import controller functions conveniently from the package.
"""

from .chunking_documents import chunking_docs
from .unique_filename_generator import generate_unique_filename
from .text_operation import TextCleaner
from .ex_images_pdf import ExtractionImagesFromPDF
from .ocr import AdvancedOCRProcessor
from .life_span import lifespan, path_rag

__all__ = [
    "chunking_docs",
    "generate_unique_filename",
    "TextCleaner",
    "ExtractionImagesFromPDF",
    "AdvancedOCRProcessor",
    "lifespan"
]
