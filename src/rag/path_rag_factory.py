
import logging
import os
import sys
import time
import numpy as np

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
            cache_size=settings.DEV_CACHE_SIZE_MB,
            memory_limit_gb=settings.DEV_MEMORY_LIMIT_GB,
            max_graph_size=settings.DEV_MAX_GRAPH_NODES,
            decay_rate=getattr(settings, "DEV_DECAY_RATE", 0.8),
            prune_thresh=getattr(settings, "DEV_PRUNE_THRESH", 0.5),
            sim_threshold=getattr(settings, "DEV_SIM_THRESHOLD", 0.7),
            checkpoint_interval=settings.DEV_CHECKPOINT_INTERVAL_DOCS,
            enable_caching=settings.DEV_ENABLE_PROFILING,
        )
        return PathRAG(embedding_model, config)

    @staticmethod
    def create_production_instance(embedding_model) -> PathRAG:
        """Create PathRAG instance optimized for production."""
        config = PathRAGConfig(
            max_workers=settings.PROD_MAX_WORKERS,
            batch_size=settings.PROD_BATCH_SIZE,
            cache_size=settings.PROD_CACHE_SIZE_MB,
            memory_limit_gb=settings.PROD_MEMORY_LIMIT_GB,
            max_graph_size=settings.PROD_MAX_GRAPH_NODES,
            decay_rate=getattr(settings, "PROD_DECAY_RATE", 0.9),
            prune_thresh=getattr(settings, "PROD_PRUNE_THRESH", 0.6),
            sim_threshold=getattr(settings, "PROD_SIM_THRESHOLD", 0.75),
            checkpoint_interval=settings.PROD_CHECKPOINT_INTERVAL_DOCS,
            enable_caching=settings.PROD_ENABLE_PROFILING,
        )
        return PathRAG(embedding_model, config)

    @staticmethod
    def create_memory_optimized_instance(embedding_model) -> PathRAG:
        """Create PathRAG instance optimized for low memory usage."""
        config = PathRAGConfig(
            max_workers=settings.MEMORY_MAX_WORKERS,
            batch_size=settings.MEMORY_BATCH_SIZE,
            cache_size=settings.MEMORY_CACHE_SIZE_MB,
            memory_limit_gb=settings.MEMORY_MEMORY_LIMIT_GB,
            max_graph_size=settings.MEMORY_MAX_GRAPH_NODES,
            decay_rate=getattr(settings, "MEMORY_DECAY_RATE", 0.7),
            prune_thresh=getattr(settings, "MEMORY_PRUNE_THRESH", 0.4),
            sim_threshold=getattr(settings, "MEMORY_SIM_THRESHOLD", 0.65),
            checkpoint_interval=settings.MEMORY_CHECKPOINT_INTERVAL_DOCS,
            enable_caching=not settings.MEMORY_ENABLE_SWAP,
        )
        return PathRAG(embedding_model, config)

if __name__ == "__main__":
    """
    Example usage of PathRAGFactory with HuggingFace embeddings.

    Demonstrates:
    1. Initializing the embedding model.
    2. Creating a PathRAG instance.
    3. Generating embeddings for example chunks.
    4. Building the semantic graph.
    5. Building FAISS index.
    6. Saving graph and displaying metrics.
    """

    import time
    from src.llms_providers import HuggingFaceModel
    from src.utils import AutoSave, checkpoint_callback

    try:
        # 1. Initialize embedding model
        embedding_model = HuggingFaceModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("HuggingFace embedding model initialized")

        # 2. Example document chunks
        chunks = [
            "Climate change is causing global temperatures to rise significantly.",
            "Rising sea levels threaten coastal cities around the world.",
            "Renewable energy sources like solar and wind are becoming more efficient.",
            "Artificial intelligence is transforming various industries rapidly.",
            "Machine learning algorithms require large datasets for training.",
            "Deep learning models can process complex patterns in data.",
            "Electric vehicles are reducing carbon emissions in transportation.",
            "Battery technology improvements enable longer-range electric cars.",
            "Sustainable agriculture practices help preserve soil quality.",
            "Organic farming methods reduce pesticide use in food production.",
            "Ocean acidification affects marine ecosystems negatively.",
            "Coral reefs are bleaching due to increased water temperatures.",
            "Deforestation contributes to loss of biodiversity globally.",
            "Reforestation efforts help restore damaged forest ecosystems.",
            "Urban planning must consider environmental sustainability factors."
        ]

        # 3. Generate embeddings
        embeddings = embedding_model.embed_texts(
            texts=chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if embeddings is None:
            raise ValueError("Embedding generation failed")
        logger.info("Generated embeddings for %d chunks", len(chunks))
        logger.info("Embeddings shape: %s", embeddings.shape)

        # 4. Create PathRAG instance
        rag = PathRAGFactory.create_production_instance(embedding_model)
        logger.info("PathRAG instance created with FAISS optimization")

        # 5. Build the semantic graph
        logger.info("Building semantic graph from %d chunks...", len(chunks))
        start_time = time.time()
        rag.build_graph(chunks=chunks, embeddings=embeddings, checkpoint_callback=checkpoint_callback)
        build_time = time.time() - start_time
        logger.info("Graph built in %.2f seconds", build_time)
        logger.info("Graph contains %d nodes and %d edges", rag.g.number_of_nodes(), rag.g.number_of_edges())

        # 6. Initialize AutoSave
        autosave = AutoSave(pathrag_instance=rag)
        logger.info("AutoSave initialized at %s", str(autosave.save_dir))

        # 7. Build FAISS index for fast similarity search
        logger.info("Building FAISS index...")
        faiss_start = time.time()
        rag.build_faiss_index()
        faiss_time = time.time() - faiss_start
        logger.info("FAISS index built in %.2f seconds", faiss_time)

        # 8. Display graph metrics
        metrics = rag.get_metrics()
        logger.info("Graph Statistics:")
        logger.info("  - Nodes: %d", metrics["nodes_count"])
        logger.info("  - Edges: %d", metrics["edges_count"])
        logger.info("  - Memory Usage: %.2f GB", metrics["memory_usage_gb"])
        logger.info("  - Average Degree: %.2f", metrics["avg_degree"])
        autosave.save_checkpoint()
    except Exception as e:
        logger.exception("Error occurred during PathRAG demo: %s", e)
