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
from src.llmsprovider import NERModel, GeminiLLM
from src.utils import setup_logging

logger = setup_logging()


def load_embedding_model():
    """
    Load a sentence embedding model from Hugging Face using SentenceTransformer.

    Returns:
        model (SentenceTransformer): The loaded embedding model.
    """
    try:
        logger.info("🔄 Starting embedding model loading...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.debug("Model name set to: %s", model_name)

        model = SentenceTransformer(model_name)

        logger.info("✅ Embedding model loaded successfully.")
        return model
    except Exception as e:
        logger.error("❌ Failed to load embedding model: %s", e, exc_info=True)
        raise


def main():
    """
    Run the LightRAG demo with:
    - Small CPU-friendly models
    - Document processing
    - RAG-based querying
    """
    logger.info("🚀 Starting LightRAG demo...")
    try:
        # Load models
        logger.debug("Calling load_embedding_model()...")
        embedding_model = load_embedding_model()

        logger.info("🧠 Initializing LightRAG...")
        rag = LightRAG(
            embedding_model=embedding_model,
            ner_model=NERModel(),
            llm_model=GeminiLLM()
        )
        logger.debug("LightRAG instance created.")
        from src.rag import document
        # Sample document
        logger.info("📄 Sample medical document prepared.")
        logger.debug("Document content: %s", document[:100])

        # Process document
        logger.info("⚙️ Processing document...")
        try:
            rag.process_document(document=document,
                                 chunk_size=512,
                                 chunk_overlap=20)
            logger.info("✅ Document processed successfully.")
        except Exception as e:
            logger.error("❌ Failed to process document: %s", e, exc_info=True)
            return

        # Sample queries
        queries = [
            "What is the difference between supervised and unsupervised learning?",
            "How do Transformers work in deep learning?",
            "Explain the role of self-attention in LLMs.",
            "What is Retrieval-Augmented Generation and why is it useful?",
            "How does LightRAG differ from traditional RAG?",
            "Name popular large language models and their use cases.",
            "What are CNNs and where are they applied?",
            "How does reinforcement learning train agents?"
        ]

        logger.info("🔍 Starting query loop...")
        for query in queries:
            try:
                logger.debug("Sending query: %s", query)
                response = rag.query(query)
                print(f"Q: {query}\nA: {response}\n")
                logger.info("✅ Query processed: %s", query)
            except Exception as e:
                logger.warning("⚠️ Failed to process query '%s': %s", query, e, exc_info=True)

    except Exception as e:
        logger.critical("💥 Fatal error in main: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
