"""
main.py

Main entry point for the Graph-RAG FastAPI application.

This application implements a Path-aware Retrieval-Augmented Generation (PathRAG)
system using semantic graphs. It is designed for concurrent usage and robust error
handling.

Responsibilities:
-----------------
1. Project Path Setup:
   - Dynamically adds the project root directory to sys.path for imports.

2. Logging & Settings:
   - Initializes application-wide logging and loads configuration settings.

3. Application Lifespan Management:
   - Manages startup and shutdown tasks through `lifespan` (see controllers/lifespan.py).
   - Handles MongoDB connection, model initialization, and PathRAG setup.

4. API Route Registration:
   - File upload, chunking, embedding, graph building, retrieval, chatbot interaction,
     storage management, resource monitoring, and user file management.

5. Static Files & UI:
   - Serves static assets and the main web interface.

6. Extra Endpoints:
   - `/graph`: Returns a Plotly JSON visualization of the semantic graph.
   - `/db-info`: Returns MongoDB databases, collections, and document counts.

Error Handling:
---------------
- Critical errors in imports, route registration, or startup result in process termination.
- Runtime errors in endpoints return appropriate HTTPException responses.

Author:
-------
ALRashid AlKiswane
"""

import asyncio
from contextlib import asynccontextmanager
import json
from logging import config
import os
from pathlib import Path
import sys
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
    from src.mongodb import get_mongo_client
    from src.routes import *
    from src.infra import setup_logging
    from src.rag import visualize_graph
    from src.helpers import get_settings, Settings
    from src.llms_providers import HuggingFaceModel
    from src.llms_providers import OllamaModel
    from src.mongodb import init_chatbot_collection, init_chunks_collection, init_embed_vector_collection
    from src.rag import PathRAGFactory
    from src.utils import ConcurrentModelManager, ThreadSafePathRAG
except ImportError as e:
    logging.critical("Import error during module loading: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging(name="MAIN")
app_settings: Settings = get_settings()

pathrag = None

# === FastAPI Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
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

        # === Initialize Collections ===
        try:
            init_chunks_collection(db)
            init_embed_vector_collection(db)
            init_chatbot_collection(db)
            logger.info("MongoDB collections initialized.")
        except Exception as coll_err:
            logger.critical("Failed to initialize MongoDB collections: %s", coll_err, exc_info=True)
            raise RuntimeError("Collection setup failed.") from coll_err

        # === Load LLM and Embedding Models ===
        try:
            app.state.model_manager = ConcurrentModelManager(
                OllamaModel,
                HuggingFaceModel,
                app_settings.OLLAMA_CHAT_MODEL,
                app_settings.EMBEDDING_MODEL_NAME,
            )
            app.state.llm = app.state.model_manager.get_llm_instance()
            app.state.embedding_model = app.state.model_manager.get_embedding_instance()
            logger.info("Models loaded with concurrent support.")
        except Exception as model_err:
            logger.critical("Failed to load models: %s", model_err, exc_info=True)
            raise RuntimeError("Model loading failed.") from model_err

        # === Initialize PathRAG ===
        try:
            global pathrag
            pathrag = PathRAGFactory().create_development_instance(
                embedding_model=app.state.embedding_model
            )
            app.state.path_rag = ThreadSafePathRAG(pathrag)

            graph_path = Path(app_settings.CHECKPOINT_GRAPH_FILE)
            if graph_path.exists():
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, app.state.path_rag.load_graph, graph_path
                    )
                    logger.info("Graph loaded from %s", graph_path)
                except Exception as e:
                    logger.warning("Failed to load existing graph: %s", e)
            else:
                logger.info("No saved graph found. A new one will be created as needed.")
        except Exception as rag_err:
            logger.critical("Failed to initialize PathRAG: %s", rag_err, exc_info=True)
            raise RuntimeError("PathRAG initialization failed.") from rag_err

        yield

    except Exception as startup_error:
        logger.exception("Fatal error during app startup: %s", startup_error)
        raise
    finally:
        try:
            if hasattr(app.state, 'model_manager'):
                app.state.model_manager.shutdown()
            logger.info("Application shutdown complete.")
        except Exception as cleanup_err:
            logger.error("Error during cleanup: %s", cleanup_err)

# === FastAPI Application Instance ===
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="Path-aware Retrieval-Augmented Generation system using semantic graphs with concurrency support.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Register Routes ===
try:
    app.include_router(upload_route)                 # File upload
    app.include_router(chunking_router)               # Chunking
    app.include_router(embedding_chunks_route)       # Embedding generation
    app.include_router(build_pathrag_route)          # Graph building
    app.include_router(live_retrieval_route)         # Retrieval
    app.include_router(chatbot_route)                # Chatbot
    app.include_router(storage_management_route)     # Storage management
    app.include_router(resource_monitor_router)      # System monitoring
    app.include_router(md_chunker_routes)            # Markdown chunker
    app.include_router(user_file_route)              # User file management
    logger.info("✅ All API routes registered successfully with concurrency support.")
except Exception as route_err:
    logger.critical("Failed to register API routes: %s", route_err, exc_info=True)
    sys.exit(1)

# === Serve Static Assets ===
app.mount("/static", StaticFiles(directory=f"{MAIN_DIR}/src/web"), name="static")

# === Serve Web UI ===
@app.get("/")
async def serve_ui():
    """
    Serve the main user interface (index.html).
    
    Returns:
        FileResponse: The main HTML page.
    """
    return FileResponse(f"{MAIN_DIR}/src/web/index.html")

@app.get("/graph")
async def get_graph(max_nodes: int = 100):
    """
    Returns a Plotly JSON representation of the semantic graph.

    Args:
        max_nodes (int): Maximum number of nodes to visualize.

    Returns:
        JSONResponse: Plotly figure as JSON.

    Raises:
        HTTPException: 
            - 404 if graph is empty or not initialized.
            - 500 if graph visualization fails.
    """
    try:
        # Use thread-safe graph access
        g = pathrag.g
        if g is None or g.number_of_nodes() == 0:
            logging.warning("Graph is empty or not initialized.")
            raise HTTPException(status_code=404, detail="Graph is empty or not initialized.")

        logging.info(f"Visualizing graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

        # Run visualization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        fig = await loop.run_in_executor(
            None,
            visualize_graph,
            g,
            max_nodes
        )

        return JSONResponse(content=json.loads(fig.to_json()))

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to generate graph visualization.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/db-info")
def get_db_info() -> Dict[str, Any]:
    """
    Get detailed MongoDB information:
    - All database names
    - All collections in each database
    - Document counts for each collection

    Returns:
        dict: MongoDB information structure.

    Raises:
        HTTPException:
            - 500 if MongoDB connection fails or query fails.
    """
    client = get_mongo_client()
    if not client:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

    try:
        db_info = {}
        for db_name in client.list_database_names():
            db = client[db_name]
            collections_info = {}
            for coll_name in db.list_collection_names():
                count = db[coll_name].count_documents({})
                collections_info[coll_name] = count
            db_info[db_name] = collections_info

        return {"databases": db_info}

    except Exception as e:
        logging.exception("Error fetching MongoDB info.")
        raise HTTPException(status_code=500, detail=f"Error fetching database info: {e}")
