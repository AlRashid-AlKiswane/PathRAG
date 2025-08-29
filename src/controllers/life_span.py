"""
lifespan.py

This module defines the FastAPI lifespan context manager for the Graph-RAG application. 
It handles application startup and shutdown tasks, ensuring proper initialization of 
databases, models, and the PathRAG reasoning system with concurrency and thread safety.

Responsibilities:
-----------------
1. Project Path Setup:
   - Dynamically adds the project root directory to sys.path for imports.

2. MongoDB Initialization:
   - Establishes a connection to the MongoDB database.
   - Initializes required collections for chunks, embeddings, and chatbot memory.

3. Model Loading:
   - Loads language models (LLM) and embedding models using ConcurrentModelManager.
   - Ensures thread-local model instances for concurrent usage.

4. PathRAG Setup:
   - Creates a PathRAG instance using PathRAGFactory.
   - Wraps it with ThreadSafePathRAG for safe concurrent graph operations.
   - Attempts to load an existing knowledge graph from disk.

5. Application Shutdown:
   - Cleans up thread pools and releases resources gracefully.

Usage:
------
In your FastAPI app:
    from lifespan import lifespan
    app = FastAPI(lifespan=lifespan)

Configuration:
--------------
- Uses values from `Settings` (via `get_settings()`).
- Logs all critical steps using the centralized logging system.

Raises:
-------
RuntimeError:
    - If MongoDB connection or collection setup fails.
    - If model initialization fails.
    - If PathRAG cannot be initialized.

Author:
-------
ALRashid AlKiswane
"""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import logging

from fastapi import FastAPI

# === Configure Project Path ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Internal Imports ===
try:
    from src.mongodb import (
        get_mongo_client,
        init_chatbot_collection,
        init_chunks_collection,
        init_embed_vector_collection,
    )
    from src.routes import *
    from src.infra import setup_logging
    from src.llms_providers import OllamaModel, HuggingFaceModel
    from src.helpers import get_settings, Settings
    from src.utils import ConcurrentModelManager, ThreadSafePathRAG
    from src.rag import PathRAGFactory
except ImportError as e:
    logging.critical("Import error during module loading: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging(name="LIFESPAN")
config: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for app startup and shutdown.

    On startup:
    - Connects to MongoDB and initializes collections.
    - Loads language and embedding models with concurrent support.
    - Sets up the PathRAG reasoning system with thread safety.

    On shutdown:
    - Cleans up thread pools and MongoDB connections.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None

    Raises:
        RuntimeError: On failure of any critical component (DB, models, PathRAG).
    """
    try:
        logger.info("Starting up Graph-RAG API with concurrency support...")

        # === MongoDB Initialization ===
        try:
            client = get_mongo_client()
            db = client["PathRAG-MongoDB"]
            app.state.db = db
            logger.info("Connected to MongoDB: %s", db.name)
        except Exception as mongo_err:
            logger.critical("Failed to connect to MongoDB: %s", mongo_err, exc_info=True)
            raise RuntimeError("Database initialization failed.") from mongo_err

        # === Initialize MongoDB Collections ===
        try:
            init_chunks_collection(db)
            init_embed_vector_collection(db)
            init_chatbot_collection(db)
            logger.info("MongoDB collections initialized.")
        except Exception as coll_err:
            logger.critical("Failed to initialize MongoDB collections: %s", coll_err, exc_info=True)
            raise RuntimeError("Collection setup failed.") from coll_err

        # === Load Models with Concurrency Support ===
        try:
            app.state.model_manager = ConcurrentModelManager(
                OllamaModel, 
                HuggingFaceModel,
                config.OLLAMA_MODEL,
                config.EMBEDDING_MODEL
            )

            # Create primary instances for backward compatibility
            app.state.llm = app.state.model_manager.get_llm_instance()
            app.state.embedding_model = app.state.model_manager.get_embedding_instance()
            
            logger.info("LLM and embedding models loaded with concurrent support.")
        except Exception as model_err:
            logger.critical("Failed to load models: %s", model_err, exc_info=True)
            raise RuntimeError("Model loading failed.") from model_err

        # === Initialize PathRAG with Thread Safety ===
        try:
            path_rag_instance = PathRAGFactory().create_development_instance(
                embedding_model=app.state.embedding_model
            )

            # Wrap with thread-safe wrapper
            global path_rag
            path_rag = ThreadSafePathRAG(path_rag_instance)
            app.state.path_rag = path_rag

            # Attempt to load graph from storage if it exists
            graph_path = Path(config.STORAGE_GRAPH)
            if graph_path.exists():
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, path_rag.load_graph, graph_path
                    )
                    logger.info("Graph loaded successfully from %s", graph_path)
                except Exception as e:
                    logger.error("Failed to load graph from %s: %s", graph_path, e)
            else:
                logger.info("No existing graph found - will create a new one when needed")

            logger.info("PathRAG initialized with thread safety.")
        except Exception as rag_err:
            logger.critical("Failed to initialize PathRAG: %s", rag_err, exc_info=True)
            raise RuntimeError("PathRAG initialization failed.") from rag_err

        yield  # Application runs here

    except Exception as startup_error:
        logger.exception("Fatal error during app startup: %s", startup_error)
        raise

    finally:
        # === Cleanup ===
        try:
            if hasattr(app.state, 'model_manager'):
                app.state.model_manager.shutdown()
            logger.info("Application shutdown complete.")
        except Exception as cleanup_err:
            logger.error("Error during cleanup: %s", cleanup_err)
