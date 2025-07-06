"""
Document Processing Pipeline

A comprehensive text processing system that transforms documents into structured chunks with embeddings.
Supports multiple input formats including raw text, text files, and PDF documents.

Key Features:
- Multi-format input handling (text, txt, pdf)
- Semantic chunking with configurable parameters
- Dense vector embeddings generation
- Metadata preservation across processing stages
- Robust error handling and logging

Typical Workflow:
1. Initialize processor with desired settings
2. Feed documents (text or files)
3. Receive structured chunks with embeddings
4. Utilize in downstream NLP applications
"""

# pylint: disable=redefined-outer-name

import logging
import os
import sys
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Additional imports for file type handling
import mimetypes
from PyPDF2 import PdfReader

# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.utils import setup_logging

logger = setup_logging()

class DocumentProcessor:
    """
    Core document processing engine that transforms input documents into analyzable chunks.

    Capabilities:
    - Processes text directly or from files (txt/pdf)
    - Splits content while preserving semantic context
    - Generates embeddings using state-of-the-art models
    - Maintains comprehensive metadata throughout

    Input Flexibility:
    - Direct text strings
    - Text files (.txt)
    - PDF documents (.pdf)

    Output Structure:
    - List of dictionaries containing:
      * Text content
      * Metadata (source, dimensions, etc.)
      * Embedding vectors

    Example:
        >>> processor = DocumentProcessor()
        >>> results = processor.process("document.pdf")
        >>> # Returns list of chunks with text/metadata/embeddings
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 embedding_model: SentenceTransformer = None):
        """
        Initialize with processing parameters.
        """
        logger.info("Initializing DocumentProcessor with chunk_size=%d, chunk_overlap=%d", 
                   chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            self.embedding_model: SentenceTransformer = embedding_model
        except Exception as e:
            logger.critical("Initialization failed: %s", str(e))
            raise

    def process(self, input_source: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """
        Main processing method that handles all input types.
        """
        try:
            # Handle direct text input
            if isinstance(input_source, str) and not os.path.isfile(input_source):
                logger.info("Processing direct text input")
                return self._process_text(input_source)

            # Handle file path input
            if isinstance(input_source, (str, Path)):
                file_path = Path(input_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                file_type = self._detect_file_type(file_path)
                logger.info("Processing %s file: %s", file_type, file_path)

                if file_type == 'text/plain':
                    return self._process_text_file(file_path)
                elif file_type == 'application/pdf':
                    return self._process_pdf_file(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")

            raise ValueError("Invalid input type - must be text, file path, or bytes")
        except Exception as e:
            logger.error("Processing failed: %s", str(e))
            raise

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type using extension and mimetypes."""
        # First check extension against supported types
        ext = file_path.suffix.lower()
        if ext == '.txt':
            return 'text/plain'
        if ext == '.pdf':
            return 'application/pdf'

        # Fall back to mimetype detection
        mime, _ = mimetypes.guess_type(file_path)
        if mime in {'text/plain', 'application/pdf'}:
            return mime

        raise ValueError(f"Unsupported file type for {file_path}")

    def _process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = {
                "source": str(file_path),
                "file_type": "text/plain",
                "file_size": os.path.getsize(file_path)
            }
            return self._process_text(text, metadata)
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, trying latin-1 as fallback")
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return self._process_text(text, metadata)

    def _process_pdf_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF file and extract text."""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            metadata = {
                "source": str(file_path),
                "file_type": "application/pdf",
                "page_count": len(pdf_reader.pages),
                "file_size": os.path.getsize(file_path)
            }
            return self._process_text(text, metadata)
        except Exception as e:
            logger.error("PDF processing failed: %s", str(e))
            raise

    def _process_text(self,
                       text: str,
                         base_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Core text processing with chunking and embedding."""
        try:
            chunks = self.text_splitter.create_documents([text])
            results = []

            for chunk in chunks:
                try:
                    embedding = self._generate_embedding(chunk.page_content)
                    results.append({
                        "text": chunk.page_content,
                        "metadata": {
                            **(base_metadata or {}),
                            "chunk_length": len(chunk.page_content),
                            "chunk_id": len(results) + 1
                        },
                        "embedding": embedding
                    })
                except Exception as e:
                    logger.warning("Chunk processing failed: %s", str(e))
                    continue

            return results
        except Exception as e:
            logger.error("Text processing failed: %s", str(e))
            raise

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error("Embedding generation failed: %s", str(e))
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    processor = DocumentProcessor()

    # Example 1: Direct text processing
    print("\n=== Text Input Processing ===")
    try:
        text = "This is a sample text to demonstrate the document processor."
        chunks = processor.process(text)
        print(f"Generated {len(chunks)} chunks from text input")
    except Exception as e:
        logging.error("Text processing failed: %s", str(e))
    
    # Example 2: PDF file processing
    print("\n=== PDF File Processing ===")
    try:
        pdf_path = "/home/alrashid/Videos/B/0ec4474b84d3e1e93f22e703b8e7fabe48f0ec1e1eea98fd33a813fee59cb473.pdf"
        if os.path.exists(pdf_path):
            chunks = processor.process(pdf_path)
            print(f"Generated {len(chunks)} chunks from PDF")
            if chunks:
                print("First chunk metadata:", chunks[0]['metadata'])
        else:
            print("PDF file not found, skipping test")
    except Exception as e:
        logging.error("PDF processing failed: %s", str(e))
