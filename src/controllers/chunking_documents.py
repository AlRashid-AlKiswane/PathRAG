"""
Document Processing Module

This module provides functionality to load and chunk documents (PDF and text) using LangChain.
It supports loading documents, extracting metadata, and splitting text into manageable chunks
for use in vector databases or LLM pipelines.

Features:
- Supports `.pdf` and `.txt` file formats
- Configurable chunking parameters via environment settings
- Lazy logging and comprehensive error handling
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup main directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.getLogger(__name__).error("Failed to set main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings

logger = setup_logging()
app_settings: Settings = get_settings()

def chunking_docs(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and chunk a document into smaller pieces using LangChain's RecursiveCharacterTextSplitter.

    Parameters:
        file_path (str): Path to the document (PDF or text file)

    Returns:
        Dict[str, Any]: A dictionary with keys:
            - 'chunks': List of chunked documents
            - 'total_chunks': Total number of chunks
            - 'file_path': Original file path

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file type is unsupported
        RuntimeError: If the document loading or chunking fails
    """
    if not file_path:
        logger.error("No file path provided.")
        raise ValueError("File path is required.")

    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    if not file_path_obj.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    if extension not in app_settings.FILE_TYPES:
        logger.warning("Unsupported file extension: %s", extension)
        raise ValueError(f"Unsupported file type: {extension}")

    loader = None
    try:
        if extension == ".pdf":
            loader = PyMuPDFLoader(file_path=file_path)
            logger.debug("Using PyMuPDFLoader for PDF file: %s", file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path=file_path)
            logger.debug("Using TextLoader for text file: %s", file_path)
    except Exception as e:
        logger.error("Failed to initialize loader for file %s: %s", file_path, e)
        raise RuntimeError(f"Failed to initialize loader: {e}") from e

    try:
        documents = loader.load()
        logger.info("Loaded %d document(s) from: %s", len(documents), file_path)
    except Exception as e:
        logger.error("Failed to load document: %s", e)
        raise RuntimeError(f"Document loading failed: {e}") from e

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_settings.CHUNKS_SIZE,
            chunk_overlap=app_settings.CHUNKS_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        logger.info("Successfully split document into %d chunks.", len(chunks))
    except Exception as e:
        logger.error("Failed to split document into chunks: %s", e)
        raise RuntimeError(f"Chunking failed: {e}") from e

    return {
        "file_path": file_path,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

if __name__ == "__main__":
    """
    Example usage for testing the document chunking module.

    Make sure to:
    - Set the appropriate environment variables or `.env` file.
    - Place a `.pdf` or `.txt` file in a known path.
    """

    import pprint

    # Example file path - replace with an actual PDF or TXT file path
    test_file_path = "/home/alrashid/Desktop/PathRAG-LightRAG/assets/docs/FRP-Database/feduc_09_1430729_20250707_202039_fcc8525e.pdf"

    try:
        result = chunking_docs(test_file_path)
        pprint.pprint({
            "File": result["file_path"],
            "Total Chunks": result["total_chunks"],
            "First Chunk": result["chunks"][0].page_content if result["chunks"] else "No chunks found"
        })
    except Exception as e:
        logger.exception("An error occurred while chunking the document: %s", e)
