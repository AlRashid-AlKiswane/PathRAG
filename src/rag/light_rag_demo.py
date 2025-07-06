"""
This module demonstrates a lightweight Retrieval-Augmented Generation (RAG) pipeline
using efficient Hugging Face models suitable for CPU environments.

Features:
- Loads a small sentence-transformer model for embeddings
- Initializes a LightRAG system with:
    - Embedding model
    - Named Entity Recognition (NER) model
    - Language model (LLM)
- Processes a sample document related to cardiology
- Answers natural language queries based on the document

Intended for demonstration, debugging, and testing RAG components in a minimal setup.

Modules Required:
- torch
- transformers
- src.rag.LightRAG
- src.llmsprovider.{NERModel, GeminiLLM}
- src.utils.setup_logging
"""

import logging
import os
import sys

# Setup project base path
# pylint: disable=logging-too-few-args
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
    logging.debug("Project base path set to: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical(
        "[Startup Critical] Failed to set up project base path. "
        "Error: %s. System paths: %s", e,
        exc_info=True
    )
    sys.exit(1)

from sentence_transformers import SentenceTransformer

from src.rag import LightRAG
from src.llmsprovider import NERModel, OllamaModel
from src.utils import setup_logging

logger = setup_logging()


from pathlib import Path
from light_rag import LightRAG

def run_case_example():
    # Initialize system
    rag = LightRAG()

    # Ingest the document
    document_path = Path("/home/alrashid/Desktop/PathRAG-LightRAG/asesst/sample.txt")
    entities_count, relations_count = rag.ingest_document(document_path)
    print(f"Ingested Document - Entities: {entities_count}, Relations: {relations_count}")

    # Ask a question
    question = "Who was Albert Einstein?"
    result = rag.query(question)

    # Print result
    print("\n--- Query Result ---")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['response']}")

    print("\nTop Chunks:")
    for chunk in result['top_chunks']:
        print(f"- {chunk.strip()[:80]}...")

    print("\nTop Entities:")
    for entity in result['top_entities']:
        print(f"- {entity}")

if __name__ == "__main__":
    run_case_example()
