
import logging
import os
import sys

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.schemas import PathRAGConfig
from src.rag.pathrag import  PathRAG
from src.helpers import get_settings, Settings

logger = setup_logging(name="PATH-RAG")
settings: Settings = get_settings()

class PathRAGFactory:
    """Factory class for creating PathRAG instances with environment-driven configurations."""

    @staticmethod
    def create_development_instance(embedding_model) -> PathRAG:
        """Create PathRAG instance optimized for development."""
        config = PathRAGConfig(
            max_workers=settings.DEV_MAX_WORKERS,
            batch_size=settings.DEV_BATCH_SIZE,
            cache_size=settings.DEV_CACHE_SIZE,
            memory_limit_gb=settings.DEV_MEMORY_LIMIT_GB,
            max_graph_size=settings.DEV_MAX_GRAPH_SIZE,
            decay_rate=settings.DEV_DECAY_RATE,
            prune_thresh=settings.DEV_PRUNE_THRESH,
            sim_threshold=settings.DEV_SIM_THRESHOLD,
            checkpoint_interval=settings.DEV_CHECKPOINT_INTERVAL,
            enable_caching=settings.DEV_ENABLE_CACHING,
        )
        return PathRAG(embedding_model, config)

    @staticmethod
    def create_production_instance(embedding_model) -> PathRAG:
        """Create PathRAG instance optimized for production."""
        config = PathRAGConfig(
            max_workers=settings.PROD_MAX_WORKERS,
            batch_size=settings.PROD_BATCH_SIZE,
            cache_size=settings.PROD_CACHE_SIZE,
            memory_limit_gb=settings.PROD_MEMORY_LIMIT_GB,
            max_graph_size=settings.PROD_MAX_GRAPH_SIZE,
            decay_rate=settings.PROD_DECAY_RATE,
            prune_thresh=settings.PROD_PRUNE_THRESH,
            sim_threshold=settings.PROD_SIM_THRESHOLD,
            checkpoint_interval=settings.PROD_CHECKPOINT_INTERVAL,
            enable_caching=settings.PROD_ENABLE_CACHING,
        )
        return PathRAG(embedding_model, config)

    @staticmethod
    def create_memory_optimized_instance(embedding_model) -> PathRAG:
        """Create PathRAG instance optimized for low memory usage."""
        config = PathRAGConfig(
            max_workers=settings.MEMORY_MAX_WORKERS,
            batch_size=settings.MEMORY_BATCH_SIZE,
            cache_size=settings.MEMORY_CACHE_SIZE,
            memory_limit_gb=settings.MEMORY_MEMORY_LIMIT_GB,
            max_graph_size=settings.MEMORY_MAX_GRAPH_SIZE,
            decay_rate=settings.MEMORY_DECAY_RATE,
            prune_thresh=settings.MEMORY_PRUNE_THRESH,
            sim_threshold=settings.MEMORY_SIM_THRESHOLD,
            checkpoint_interval=settings.MEMORY_CHECKPOINT_INTERVAL,
            enable_caching=settings.MEMORY_ENABLE_CACHING,
        )
        return PathRAG(embedding_model, config)

if __name__ == "__main__":
    """
    Example usage of PathRAGFactory with HuggingFace embeddings.

    This script demonstrates:
    1. Initializing the embedding model.
    2. Creating a memory-optimized PathRAG instance.
    3. Generating embeddings for example texts.
    4. Building the semantic graph from chunks and embeddings.
    5. Saving the resulting graph to disk.

    Notes
    -----
    - Replace the placeholder texts with your real document chunks.
    - The saved graph can later be reloaded for retrieval and reasoning.
    """

    import logging
    from src.llms_providers import HuggingFaceModel
    from pathlib import Path
    from src.rag.plot_graph import visualize_graph
    
    try:
        # 1. Initialize embedding model
        embedding_model = HuggingFaceModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✅ HuggingFace embedding model initialized")

        # 2. Create a production PathRAG instance
        rag = PathRAGFactory.create_production_instance(embedding_model)
        logger.info("✅ PathRAG production instance created")

        # 3. Example document chunks
        chunks = [
            "Graph neural networks extend deep learning to structured data.",
            "Retrieval-Augmented Generation combines retrieval with LLM reasoning.",
            "FastAPI is a modern web framework for building APIs in Python."
        ]

        # 4. Generate embeddings
        embeddings = embedding_model.embed_texts(texts=chunks)
        if embeddings is None:
            raise ValueError("Embedding generation failed")
        logger.info("✅ Generated embeddings for %d chunks", len(chunks))

        # 5. Build the semantic graph
        rag.build_graph(chunks=chunks, embeddings=embeddings)
        logger.info(
            "✅ Semantic graph built with %d nodes and %d edges",
            rag.g.number_of_nodes(),
            rag.g.number_of_edges()
        )

        # 6. Save the graph
        output_path = Path("pathrag_data/pathrag_graph.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rag.save_graph(file_path=str(output_path))
        logger.info("✅ Graph saved to %s", output_path)

        # 7. Visualize the graph (Plotly figure)
        fig = visualize_graph(g=rag.g, max_nodes=5)
        fig.show()

    except Exception as e:
        logger.exception("❌ Error occurred while running PathRAG demo: %s", e)
