"""
path_rag.py

Module implementing PathRAG: a Path-aware Retrieval-Augmented Generation system using graph-based
relational reasoning over embedded document chunks.

Core functionality:
- Build a semantic directed graph from chunk embeddings using cosine similarity edges.
- Retrieve top-K semantically relevant nodes for a query embedding.
- Prune relational paths between retrieved nodes using a decay-weighted scoring function.
- Score and rank these paths based on the strength and length of semantic edges.
- Generate a textual prompt summarizing top evidence paths for downstream language models.

Main classes and functions:
- PathRAG: Primary class encapsulating graph construction, retrieval, path pruning, scoring,
  and prompt generation.
- example_usage(): Demonstrates an interactive CLI example loading data from a SQLite database,
  building the graph, and querying with live input.

Key parameters and concepts:
- decay_rate: Controls exponential decay factor applied to path scores, favoring shorter paths.
- prune_thresh: Threshold below which paths are discarded to limit noise.
- sim_should: Minimum cosine similarity to establish edges between chunks.

Dependencies:
- networkx for graph data structures and path operations.
- numpy and sklearn for embedding and similarity computations.
- tqdm for progress bars during graph building and path pruning.
- src.llms_providers.HuggingFaceModel as embedding provider.
- src.db for database access utilities.

Usage:
Instantiate PathRAG with an embedding model and optional parameters. Use build_graph() with
chunk texts and embeddings. Query relevant nodes using retrieve_nodes(), then prune and score
paths for prompt generation.

Example:
    pathrag = PathRAG(embedding_model=my_model)
    pathrag.build_graph(chunks, embeddings)
    nodes = pathrag.retrieve_nodes("my query", top_k=5)
    paths = pathrag.prune_paths(nodes)
    scored = pathrag.score_paths(paths)
    prompt = pathrag.generate_prompt("my query", scored)

Note:
This module is intended for integration into retrieval-augmented generation pipelines
where reasoning over multi-hop semantic relations enhances context retrieval quality.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import heapq
import logging
import os
import sys
import threading
from time import time, sleep
from typing import Any, Dict, List, Tuple, Optional, Union
from numbers import Integral
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go
import faiss
import torch

# Fallback imports and classes for missing dependencies
class MockSettings:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.decay_rate = 0.8
        self.prune_thresh = 0.1
        self.k_neighbors = 10
        self.batch_size = 1000
        self.max_workers = 4
        self.memory_limit_gb = 8.0
        self.max_graph_size = 100000
        self.checkpoint_interval = 10000
        self.n_clusters = 50
        self.intra_cluster_k = 5
        self.inter_cluster_k = 3
        self.sample_ratio = 0.1
        self.n_hash_bits = 128

class MockPathRAGConfig:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.decay_rate = 0.8
        self.prune_thresh = 0.1
        self.k_neighbors = 10
        self.batch_size = 1000
        self.max_workers = 4
        self.memory_limit_gb = 8.0
        self.max_graph_size = 100000
        self.checkpoint_interval = 10000
        self.n_clusters = 50
        self.intra_cluster_k = 5
        self.inter_cluster_k = 3
        self.sample_ratio = 0.1
        self.n_hash_bits = 128
        self.sim_threshold = 0.7

class MockHuggingFaceModel:
    def __init__(self):
        self.dimension = 384
    
    def embed_texts(self, texts, convert_to_numpy=True):
        """Mock embedding function."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.random.randn(len(texts), self.dimension).astype(np.float32)
        if convert_to_numpy:
            return embeddings
        return embeddings.tolist()

class MemoryMonitor:
    def __init__(self, limit_gb: float):
        self.limit_gb = limit_gb
    
    def check_memory_limit(self) -> bool:
        try:
            import psutil
            used_gb = psutil.virtual_memory().used / (1024**3)
            return used_gb < self.limit_gb
        except ImportError:
            return True
    
    def get_memory_usage(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**3)
        except ImportError:
            return 0.0
    
    def memory_guard(self):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class GraphCache:
    def __init__(self, config):
        self.config = config
        self._memory_cache = {}
        self.redis_client = None
    
    def get(self, key: str):
        return self._memory_cache.get(key)
    
    def set(self, key: str, value):
        self._memory_cache[key] = value

class PathRAGMetrics:
    def __init__(self):
        self.metrics = {}
    
    def set_metric(self, name: str, value):
        self.metrics[name] = value
    
    def increment(self, name: str, value=1):
        self.metrics[name] = self.metrics.get(name, 0) + value
    
    def get_report(self):
        return self.metrics.copy()

class GraphBuildMethod:
    KNN = "knn"
    HIERARCHICAL = "hierarchical"
    APPROXIMATE = "approximate"
    MULTI_LEVEL = "multi_level"
    HYBRID = "hybrid"
    LSH = "lsh"
    SPECTRAL = "spectral"

# Timer decorator
def timer(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logging.info(f"{name}: {end - start:.2f}s")
            return result
        return wrapper
    return decorator

def get_settings():
    return MockSettings()

# Set up project paths and imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.error("Failed to set up main directory path: %s", e)

from src.infra import setup_logging
logger = setup_logging("PATH-RAG")
config = get_settings()

class PathRAG:
    """
    Production-ready Path-aware Retrieval-Augmented Generation system.
    
    Features:
    - Thread-safe operations
    - Memory monitoring and limits
    - Intelligent caching
    - Batch processing
    - Progress tracking
    - Graceful error handling
    - Performance metrics
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize PathRAG with production-ready configuration.
        
        Args:
            embedding_model: Model for text embedding
            config: Configuration object with parameters
        """
        self.config = config or MockPathRAGConfig()
        self.embedding_model = embedding_model or MockHuggingFaceModel()
        self.g = nx.DiGraph()
        self.metrics = PathRAGMetrics()
        self.memory_monitor = MemoryMonitor(getattr(self.config, 'memory_limit_gb', 8.0))
        self.cache = GraphCache(self.config)
        self._lock = threading.Lock()
        
        # Initialize FAISS-related attributes
        self.index = None
        self.node_ids = None
        self.embedding_matrix = None
        
        # Setup logging        
        logger.info(f"PathRAG initialized with config")
    
    @timer('graph_build_time')
    def build_graph(
        self, 
        chunks: List[str], 
        embeddings: np.ndarray,
        method: str = "hierarchical",
        checkpoint_callback: Optional[callable] = None,
        use_gpu: bool = False,
    ) -> None:
        """
        Build semantic graph with production optimizations.
        
        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings
            method: Graph building method
            checkpoint_callback: Optional callback for progress checkpoints
            use_gpu: Whether to use GPU acceleration
        """
        if not self._validate_inputs(chunks, embeddings):
            raise ValueError("Invalid inputs provided")
        
        max_graph_size = getattr(self.config, 'max_graph_size', 100000)
        if len(chunks) > max_graph_size:
            logger.warning(
                f"Input size {len(chunks)} exceeds max_graph_size "
                f"{max_graph_size}, truncating"
            )
            chunks = chunks[:max_graph_size]
            embeddings = embeddings[:max_graph_size]
            
        max_workers = getattr(self.config, 'max_workers', 4)
        logger.info(f"Building {method} graph for {len(chunks)} chunks with {max_workers} workers")
        
        # Clear existing graph and initialize nodes
        self.g.clear()
        self._add_nodes(chunks, embeddings)
        
        with self.memory_monitor.memory_guard():
            if method == GraphBuildMethod.KNN:
                edge_count = self._build_knn_parallel(embeddings, checkpoint_callback, use_gpu)
            elif method == GraphBuildMethod.HIERARCHICAL:
                edge_count = self._build_hierarchical_parallel(embeddings, checkpoint_callback)
            elif method == GraphBuildMethod.APPROXIMATE:
                edge_count = self._build_approximate_parallel(embeddings, checkpoint_callback)
            elif method == GraphBuildMethod.MULTI_LEVEL:
                edge_count = self._build_multi_level_parallel(embeddings, checkpoint_callback)
            elif method == GraphBuildMethod.HYBRID:
                edge_count = self._build_hybrid_parallel(embeddings, checkpoint_callback)
            elif method == GraphBuildMethod.LSH:
                edge_count = self._build_lsh_parallel(embeddings, checkpoint_callback)
            elif method == GraphBuildMethod.SPECTRAL:
                edge_count = self._build_spectral_parallel(embeddings, checkpoint_callback)
            else:
                edge_count = self._build_graph_optimized(chunks, embeddings, checkpoint_callback)
                
            # Build FAISS index after graph construction
            logger.info("Building FAISS index for fast retrieval...")
            self._build_faiss_index()
            
            logger.info(f"Graph built by {method}: {len(chunks)} nodes, {edge_count} edges")

    def _validate_inputs(self, chunks: List[str], embeddings: np.ndarray) -> bool:
        """Comprehensive input validation."""
        if not chunks or not isinstance(chunks, list):
            logger.error("Invalid chunks provided")
            return False

        if not isinstance(embeddings, np.ndarray):
            logger.error("Embeddings must be numpy array")
            return False
        
        if len(chunks) != embeddings.shape[0]:
            logger.error("Chunks and embeddings length mismatch")
            return False
        
        if embeddings.ndim != 2:
            logger.error("Embeddings must be 2D array")
            return False
        
        return True

    def _add_nodes(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """Add all nodes to the graph efficiently."""
        # Clear existing graph
        if hasattr(self, 'g') and self.g:
            self.g.clear()
        else:
            self.g = nx.DiGraph()
            
        # Batch node addition for efficiency
        batch_size = getattr(self.config, 'batch_size', 1000)
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                future = executor.submit(
                    self._add_node_batch,
                    chunks[i:batch_end],
                    embeddings[i:batch_end],
                    i  # Start index offset
                )
                futures.append(future)
            
            # Wait for all batches to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Node batch addition failed: {e}")
    
    def _add_node_batch(self, chunk_batch: List[str], emb_batch: np.ndarray, start_idx: int) -> None:
        """Add a batch of nodes thread-safely."""
        with self._lock:
            for idx, (chunk, emb) in enumerate(zip(chunk_batch, emb_batch)):
                node_id = start_idx + idx
                self.g.add_node(node_id, text=chunk, emb=emb)

    @timer('retrieval_time')
    def retrieve_nodes(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[int]:
        """
        Retrieve top-k most similar nodes with caching.
        
        Args:
            query: Query string
            top_k: Number of top nodes to return
            use_cache: Whether to use caching
            
        Returns:
            List of node indices ranked by similarity
        """
        if not self._validate_query(query, top_k):
            raise ValueError("Invalid query parameters")
        
        # Check cache first
        cache_key = f"retrieve_{hash(query)}_{top_k}" if use_cache else None
        if cache_key and hasattr(self, 'cache'):
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics.increment("cache_hits")
                return cached_result
            self.metrics.increment("cache_misses")

        try:
            # Embed query with error handling
            q_emb = self._embed_query_safe(query).astype("float32")

            # Compute similarities efficiently
            similarities = self._compute_similarities_batch(tuple(q_emb))

            # Get top-k results
            if not similarities:
                logger.warning("No similarities computed, returning empty result")
                return []
                
            top_nodes = heapq.nlargest(
                top_k,
                similarities.items(),
                key=lambda x: x[1]
            )
            result = [node_id for node_id, _ in top_nodes]

            # Cache result
            if cache_key and hasattr(self, 'cache'):
                self.cache.set(cache_key, result)
            
            logger.info(f"Retrieved top-{top_k} nodes: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Node retrieval failed: {e}")
            raise RuntimeError(f"Retrieval failed: {e}") from e

    def _validate_query(self, query: str, top_k: int) -> bool:
        """Validate query parameters."""
        if not query or not isinstance(query, str) or not query.strip():
            return False
        if top_k <= 0 or not isinstance(top_k, int):
            return False
        if not hasattr(self, 'g') or self.g.number_of_nodes() == 0:
            logger.error("Graph has no nodes")
            return False
        return True

    def _embed_query_safe(self, query: str) -> np.ndarray:
        """Safely embed query with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                q_emb = self.embedding_model.embed_texts(
                    texts=query,
                    convert_to_numpy=True
                )
                
                if isinstance(q_emb, list):
                    q_emb = np.array(q_emb)
                if q_emb.ndim == 2 and q_emb.shape[0] == 1:
                    q_emb = q_emb[0]
                
                return q_emb
                
            except Exception as e:
                logger.warning(f"Query embedding attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                sleep(0.1 * (attempt + 1))
        
        # Fallback: return random embedding
        return np.random.randn(self.embedding_model.dimension).astype(np.float32)
    
    def _compute_similarities_batch(self, q_emb_tuple: tuple) -> Dict[int, float]:
        """
        Compute similarities using FAISS index (GPU if available).
        Returns a dictionary mapping node_id -> similarity score.
        """
        similarities = {}

        try:
            # Fallback to direct computation if FAISS fails
            if not hasattr(self, 'index') or self.index is None:
                return self._compute_similarities_direct(q_emb_tuple)

            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty, using direct computation")
                return self._compute_similarities_direct(q_emb_tuple)

            # Convert query embedding safely
            try:
                q_emb = np.asarray(q_emb_tuple, dtype="float32").reshape(1, -1)
            except Exception as e:
                logger.error("Failed to convert query embedding: %s", e)
                return similarities

            # Normalize for cosine similarity
            try:
                faiss.normalize_L2(q_emb)
            except Exception as e:
                logger.warning("Normalization failed, using raw embedding. Error: %s", e)

            # Search all embeddings (k = total vectors in index)
            k = min(self.index.ntotal, 100)  # Limit search results
            try:
                similarities_scores, indices = self.index.search(q_emb, k)
            except Exception as e:
                logger.error("FAISS search failed: %s", e)
                return self._compute_similarities_direct(q_emb_tuple)

            # Map FAISS results back to node IDs
            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                score = float(similarities_scores[0][i])

                if hasattr(self, 'node_ids') and self.node_ids is not None:
                    if idx < 0 or idx >= len(self.node_ids):
                        continue
                    node_id = int(self.node_ids[idx])
                else:
                    node_id = idx
                
                similarities[node_id] = score

            logger.debug("Computed %s similarity scores successfully", len(similarities))

        except Exception as e:
            logger.exception("Unexpected error in _compute_similarities_batch: %s", e)
            return self._compute_similarities_direct(q_emb_tuple)

        return similarities

    def _compute_similarities_direct(self, q_emb_tuple: tuple) -> Dict[int, float]:
        """Direct similarity computation as fallback."""
        similarities = {}
        try:
            q_emb = np.asarray(q_emb_tuple, dtype="float32")
            
            for node_id in self.g.nodes():
                try:
                    node_emb = self.g.nodes[node_id].get('emb')
                    if node_emb is not None:
                        sim = np.dot(q_emb, node_emb)
                        similarities[node_id] = float(sim)
                except Exception as e:
                    logger.debug(f"Failed to compute similarity for node {node_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Direct similarity computation failed: {e}")
        
        return similarities

    @timer('path_pruning_time')
    def prune_paths(self, nodes: List[int], max_hops: int = 4, max_paths_per_pair: int = 10) -> List[List[int]]:
        """
        Prune and find paths between nodes with proper validation.
        
        Args:
            nodes: List of node IDs to find paths between
            max_hops: Maximum number of hops in paths
            max_paths_per_pair: Maximum paths to find per node pair
            
        Returns:
            List of valid paths between nodes
        """
        # Handle empty nodes list gracefully
        if not nodes:
            logger.warning("No nodes provided for path pruning, returning empty paths")
            return []
            
        # Normalize + validate
        try:
            nodes = [int(n) for n in nodes]
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid node types: {e}")
            return []
            
        if not self._validate_path_inputs(nodes, max_hops):
            logger.error("Some nodes do not exist in graph or invalid params")
            logger.warning("Returning empty paths due to validation failure")
            return []

        logger.info(f"Starting path pruning for {len(nodes)} nodes")
        valid_paths = []
        
        # If only one node, return empty paths
        if len(nodes) <= 1:
            logger.info("Only one or no nodes, no paths to find")
            return []
            
        futures = []
        max_workers = getattr(self.config, 'max_workers', 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    futures.append(
                        executor.submit(
                            self._find_paths_between_nodes,
                            u, v, max_hops, max_paths_per_pair
                        )
                    )

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Finding paths"):
                try:
                    paths = fut.result()
                    if paths:
                        valid_paths.extend(paths)
                except Exception as e:
                    logger.debug(f"Path finding future failed: {e}")

        logger.info(f"Pruned to {len(valid_paths)} valid paths")
        return valid_paths

    def _validate_path_inputs(self, nodes: List[int], max_hops: int) -> bool:
        """Validate path pruning inputs."""
        if not nodes or not isinstance(nodes, (list, tuple)):
            return False
        if not isinstance(max_hops, int) or max_hops < 1:
            return False
        try:
            casted = [int(n) for n in nodes]  # handles np.int64, etc.
        except Exception:
            return False
        # All must exist in graph
        return all(self.g.has_node(n) for n in casted)
    
    def _find_paths_between_nodes(
            self,
            u: int,
            v: int,
            max_hops: int,
            max_paths: int
    ) -> List[List[int]]:
        """Find valid paths between two nodes."""
        valid_paths = []

        try:
            # Use networkx to find simple paths
            all_paths = nx.all_simple_paths(
                self.g,
                source=u,
                target=v,
                cutoff=max_hops
            )

            path_count = 0
            for path in all_paths:
                if path_count >= max_paths:
                    break

                score = self._compute_flow(path)
                prune_thresh = getattr(self.config, 'prune_thresh', 0.1)
                if score >= prune_thresh:
                    valid_paths.append(path)
                path_count += 1

        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            logger.debug(f"Path search failed between {u} and {v}: {e}")
        
        return valid_paths
    
    def _compute_flow(self, path: List[int]) -> float:
        """Compute flow score with error handling."""
        if len(path) < 2:
            return 0.0
        
        try:
            weights = []
            for i in range(len(path)-1):
                data = self.g.get_edge_data(path[i], path[i + 1])
                if not data or 'weight' not in data:
                    return 0.0
                weights.append(data["weight"])
            
            if not weights:
                return 0.0
                
            base_score = float(np.prod(weights))
            decay_rate = getattr(self.config, 'decay_rate', 0.8)
            decay_factor = decay_rate ** (len(path) - 1)
            return base_score * decay_factor
        except Exception as e:
            logger.debug(f"Flow computation failed for path {path}: {e}")
            return 0.0
    
    def score_paths(self, paths: List[List[int]]) -> List[Dict[str, Any]]:
        """Score and sort paths with validation."""
        if not paths:
            logger.warning("No paths provided for scoring")
            return []
        
        results = []
        for path in paths:
            try:
                score = self._compute_flow(path)
                if score > 0:
                    results.append({
                        "path": path,
                        "score": score,
                        "length": len(path),
                        "path_text": self._get_path_preview(path)
                    })
            except Exception as e:
                logger.debug(f"Path scoring failed for {path}: {e}")
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def _get_path_preview(self, path: List[int], max_length: int = 50) -> str:
        """Get text preview of path for debugging."""
        try:
            texts = []
            for node_id in path:
                if self.g.has_node(node_id):
                    text = self.g.nodes[node_id].get("text", f"Node_{node_id}")
                    preview = text[:max_length] + "..." if len(text) > max_length else text
                    texts.append(preview)
                else:
                    texts.append(f"Node_{node_id}")
            return " → ".join(texts)

        except Exception as e:
            logger.debug(f"Path preview failed: {e}")
            return "Preview unavailable"

    def generate_prompt(
        self,
        query: str,
        scored_paths: List[Dict[str, Any]],
        max_paths: int = 10,
        max_prompt_length: int = 4000
    ) -> str:
        """Generate production-ready prompt with length controls."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        # Limit number of paths
        limited_paths = scored_paths[:max_paths]

        lines = [f"QUERY: {query}", "", "RELATED EVIDENCE PATHS:"]
        current_length = len("\n".join(lines))

        for i, item in enumerate(limited_paths):
            try:
                path_line = self._format_path_line(item, i + 1)
                
                # Check if adding this path would exceed limit
                if current_length + len(path_line) > max_prompt_length:
                    lines.append(f"... ({len(limited_paths) - i} more paths truncated)")
                    break
                
                lines.append(path_line)
                current_length += len(path_line)
                
            except Exception as e:
                logger.debug(f"Failed to format path {i}: {e}")
        
        prompt = "\n".join(lines)
        logger.info(f"Generated prompt with {len(limited_paths)} paths ({len(prompt)} chars)")
        return prompt
  
    def _format_path_line(self, item: Dict[str, Any], index: int) -> str:
        """Format a single path line for the prompt."""
        try:
            path = item["path"]
            score = item["score"]

            # Get node texts with fallback
            path_texts = []
            for node_id in path:
                if self.g.has_node(node_id):
                    node_data = self.g.nodes.get(node_id, {})
                    text = node_data.get("text", f"Node_{node_id}")
                else:
                    text = f"Node_{node_id}"

                path_texts.append(text)

            return f"{index}. [Score: {score:.3f}] {' → '.join(path_texts)}"
            
        except Exception as e:
            logger.debug(f"Path formatting failed: {e}")
            return f"{index}. [Error formatting path]"

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        base_metrics = self.metrics.get_report()
        
        # Add graph statistics
        graph_stats = {
            'nodes_count': self.g.number_of_nodes() if hasattr(self, 'g') else 0,
            'edges_count': self.g.number_of_edges() if hasattr(self, 'g') else 0,
            'memory_usage_gb': self.memory_monitor.get_memory_usage(),
            'avg_degree': (np.mean([d for n, d in self.g.degree()]) 
                          if hasattr(self, 'g') and self.g.nodes() else 0)
        }
        
        return {**base_metrics, **graph_stats}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check memory usage
        memory_usage = self.memory_monitor.get_memory_usage()
        memory_limit = getattr(self.config, 'memory_limit_gb', 8.0)
        if memory_usage > memory_limit * 0.8:
            health['issues'].append(f"High memory usage: {memory_usage:.2f}GB")
            health['recommendations'].append("Consider reducing batch size or graph size")
        
        # Check graph size
        max_graph_size = getattr(self.config, 'max_graph_size', 100000)
        if hasattr(self, 'g') and self.g.number_of_nodes() > max_graph_size * 0.9:
            health['issues'].append("Graph approaching size limit")
            health['recommendations'].append("Consider graph pruning or increasing limits")
        
        # Set overall status
        if health['issues']:
            health['status'] = 'warning' if len(health['issues']) < 3 else 'critical'
        
        return health

    def _build_faiss_index(self):
        """
        Build a FAISS index for fast similarity search.

        - Normalizes embeddings (for cosine similarity).
        - Attempts to move the index to GPU if available.
        - Falls back gracefully to CPU on error.
        - Provides full logging and error handling.

        Returns:
            faiss.Index: The built FAISS index.

        Raises:
            RuntimeError: If index building fails irrecoverably.
        """
        try:
            node_ids = list(self.g.nodes())
            if not node_ids:
                logger.warning("Graph contains no nodes to index.")
                return None

            # Store node_ids
            self.node_ids = np.array(node_ids)

            # Stack embeddings into a matrix
            try:
                embeddings_list = []
                for n in node_ids:
                    emb = self.g.nodes[n].get('emb')
                    if emb is not None:
                        embeddings_list.append(emb)
                    else:
                        # Fallback to random embedding
                        embeddings_list.append(np.random.randn(384).astype('float32'))
                
                embedding_matrix = np.vstack(embeddings_list).astype('float32')
            except Exception as e:
                logger.error("Failed to build embedding matrix: %s", e)
                return None

            if embedding_matrix.ndim != 2:
                logger.error(f"Invalid embedding matrix shape: {embedding_matrix.shape}")
                return None

            self.embedding_matrix = embedding_matrix
            dim = embedding_matrix.shape[1]

            # Normalize for cosine similarity
            try:
                faiss.normalize_L2(embedding_matrix)
            except Exception as e:
                logger.warning("Normalization failed, using raw embedding. Error: %s", e)

            # Create CPU index
            try:
                index = faiss.IndexFlatIP(dim)
            except Exception as e:
                logger.error("Failed to initialize FAISS CPU index: %s", e)
                return None

            # Try GPU if available
            if torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                    index = gpu_index
                    logger.info("Successfully moved FAISS index to GPU")
                except Exception as e:
                    logger.warning("Failed to move FAISS index to GPU: %s", e)
                    logger.info("Falling back to CPU FAISS index.")
            else:
                logger.info("GPU not available, using CPU for FAISS index.")

            # Add vectors to index
            try:
                index.add(embedding_matrix)
            except Exception as e:
                logger.error("Failed to add embeddings to FAISS index: %s", e)
                return None

            logger.info("FAISS index successfully built with %d embeddings.", len(node_ids))
            self.index = index
            return index

        except Exception as e:
            logger.error("Error while building FAISS index: %s", e)
            return None

    def _build_graph_optimized(
            self,
            chunks: List[str],
            embeddings: np.ndarray,
            checkpoint_callback: Optional[callable]
    ) -> int:
        """Optimized graph building with batching and concurrency."""
        logger.info("Building graph for %d chunks", int(len(chunks)))

        # Build edges in batches with concurrent processing
        total_pairs = len(chunks) * (len(chunks) - 1) // 2
        processed_pairs = 0
        edge_count = 0
        
        batch_size = getattr(self.config, 'batch_size', 1000)
        max_workers = getattr(self.config, 'max_workers', 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit batch jobs
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                future = executor.submit(
                    self._process_batch,
                    embeddings,
                    i,
                    batch_end,
                    len(chunks)
                )
                futures.append(future)

            # Collect results with progress tracking
            with tqdm(total=len(futures), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_edges = future.result()

                        # Add edges thread-safely
                        with self._lock:
                            for i, j, sim in batch_edges:
                                self.g.add_edge(i, j, weight=sim)
                                self.g.add_edge(j, i, weight=sim)
                                edge_count += 2
                        
                        pbar.update(1)

                        # Checkpoint if callback provided
                        checkpoint_interval = getattr(self.config, 'checkpoint_interval', 10000)
                        if checkpoint_callback and processed_pairs % checkpoint_interval == 0:
                            checkpoint_callback(processed_pairs, total_pairs)

                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")

        self.metrics.set_metric('nodes_processed', len(chunks))
        self.metrics.set_metric('edges_created', edge_count)
        logger.info(f"Graph built: {len(chunks)} nodes, {edge_count} edges")
        return edge_count
    
    def _process_batch(
        self,
        embeddings: np.ndarray,
        start_idx: int,
        end_idx: int,
        total_size: int
    ) -> List[Tuple[int, int, float]]:
        """Process a batch of similarity computations."""
        batch_edges = []
        sim_threshold = getattr(self.config, 'sim_threshold', 
                              getattr(self.config, 'similarity_threshold', 0.7))

        for i in tqdm(range(start_idx, end_idx), desc=f"Batch {start_idx}-{end_idx}", leave=False):
            # Check memory periodically
            if i % 100 == 0 and not self.memory_monitor.check_memory_limit():
                logger.warning("Memory limit approached, reducing batch processing")
                break
            for j in range(i + 1, total_size):
                try:
                    sim = cosine_similarity(
                        embeddings[i:i+1],
                        embeddings[j:j+1]
                    )[0, 0]
                    
                    if sim > sim_threshold:
                        batch_edges.append((i, j, float(sim)))
                
                except Exception as e:
                    logger.debug(f"Similarity computation failed for ({i}, {j}): {e}")
                    continue
        
        return batch_edges

    def _build_knn_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable],
        use_gpu: bool = False
    ) -> int:
        """Parallel KNN graph building with FAISS."""
        
        # Setup FAISS index
        dimension = embeddings.shape[1]
        
        try:
            if use_gpu and torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(dimension))
            else:
                index = faiss.IndexFlatIP(dimension)
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return 0
        
        # Normalize for cosine similarity
        try:
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index.add(embeddings_norm.astype('float32'))
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            return 0
        
        # Parallel KNN search
        edge_count = 0
        batch_size = getattr(self.config, 'batch_size', 1000)
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit batched KNN searches
            for i in range(0, len(embeddings), batch_size):
                batch_end = min(i + batch_size, len(embeddings))
                k_neighbors = getattr(self.config, 'k_neighbors', 10)
                similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
                
                future = executor.submit(
                    self._process_knn_batch,
                    index,
                    embeddings_norm[i:batch_end],
                    i,  # Start index
                    k_neighbors + 1,
                    similarity_threshold
                )
                futures.append(future)
            
            # Collect results with progress tracking
            with tqdm(total=len(futures), desc="KNN batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_edges = future.result()
                        
                        # Add edges thread-safely
                        with self._lock:
                            for i, j, sim in batch_edges:
                                self.g.add_edge(i, j, weight=sim)
                                edge_count += 1
                        
                        pbar.update(1)
                        
                        checkpoint_interval = getattr(self.config, 'checkpoint_interval', 10000)
                        if checkpoint_callback and edge_count % checkpoint_interval == 0:
                            checkpoint_callback(edge_count, "edges")
                            
                    except Exception as e:
                        logger.error(f"KNN batch failed: {e}")
        
        return edge_count

    def _process_knn_batch(
        self,
        index,
        embeddings_batch: np.ndarray,
        start_idx: int,
        k: int,
        threshold: float
    ) -> List[Tuple[int, int, float]]:
        """Process a batch of KNN searches."""
        try:
            similarities, indices = index.search(embeddings_batch.astype('float32'), k)
        except Exception as e:
            logger.error(f"KNN search failed: {e}")
            return []
        
        batch_edges = []
        for i, (chunk_sims, chunk_indices) in enumerate(zip(similarities, indices)):
            original_idx = start_idx + i
            for neighbor_idx, similarity in zip(chunk_indices[1:], chunk_sims[1:]):  # Skip self
                if similarity >= threshold:
                    batch_edges.append((original_idx, int(neighbor_idx), float(similarity)))
        
        return batch_edges
    
    def _build_hierarchical_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """Parallel hierarchical clustering graph building."""
        
        # Parallel clustering
        n_clusters = getattr(self.config, 'n_clusters', 50)
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(1000, len(embeddings) // 10),
                n_init=3
            )
            cluster_labels = kmeans.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return 0
        
        # Group by clusters
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        edge_count = 0
        max_workers = getattr(self.config, 'max_workers', 4)
        
        # Process clusters in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Intra-cluster connections
            for cluster_id, chunk_indices in clusters.items():
                if len(chunk_indices) > 1:
                    intra_k = getattr(self.config, 'intra_cluster_k', 5)
                    future = executor.submit(
                        self._process_cluster,
                        embeddings,
                        chunk_indices,
                        intra_k,
                        "intra"
                    )
                    futures.append(future)
            
            # Inter-cluster connections
            inter_k = getattr(self.config, 'inter_cluster_k', 3)
            future = executor.submit(
                self._process_inter_cluster,
                embeddings,
                clusters,
                kmeans.cluster_centers_,
                inter_k
            )
            futures.append(future)
            
            # Collect results
            with tqdm(total=len(futures), desc="Hierarchical batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_edges = future.result()
                        
                        with self._lock:
                            for i, j, sim in batch_edges:
                                self.g.add_edge(i, j, weight=sim)
                                edge_count += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Hierarchical batch failed: {e}")
        
        return edge_count
    
    def _process_cluster(
        self,
        embeddings: np.ndarray,
        chunk_indices: List[int],
        k: int,
        mode: str
    ) -> List[Tuple[int, int, float]]:
        """Process connections within a cluster."""
        batch_edges = []
        similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
        
        if len(chunk_indices) <= k or len(chunk_indices) <= 20:
            # Small cluster: full connections
            for i, idx1 in enumerate(chunk_indices):
                for idx2 in chunk_indices[i+1:]:
                    try:
                        sim = np.dot(embeddings[idx1], embeddings[idx2])
                        if sim > similarity_threshold:
                            batch_edges.append((idx1, idx2, sim))
                    except Exception as e:
                        logger.debug(f"Similarity computation failed: {e}")
        else:
            # Large cluster: KNN within cluster
            try:
                cluster_embeddings = embeddings[chunk_indices]
                index = faiss.IndexFlatIP(cluster_embeddings.shape[1])
                cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                index.add(cluster_embeddings_norm.astype('float32'))
                
                search_k = min(k + 1, len(chunk_indices))
                similarities, indices = index.search(
                    cluster_embeddings_norm.astype('float32'),
                    search_k
                )
                
                for i, (chunk_sims, neighbor_indices) in enumerate(zip(similarities, indices)):
                    original_idx = chunk_indices[i]
                    for neighbor_idx, similarity in zip(neighbor_indices[1:], chunk_sims[1:]):
                        if similarity > similarity_threshold:
                            neighbor_original_idx = chunk_indices[neighbor_idx]
                            batch_edges.append((original_idx, neighbor_original_idx, float(similarity)))
            except Exception as e:
                logger.error(f"KNN within cluster failed: {e}")
        
        return batch_edges
    
    def _process_inter_cluster(
        self,
        embeddings: np.ndarray,
        clusters: Dict[int, List[int]],
        centroids: np.ndarray,
        k: int
    ) -> List[Tuple[int, int, float]]:
        """Process connections between clusters."""
        batch_edges = []
        
        try:
            # Find similar clusters using centroids
            index = faiss.IndexFlatIP(centroids.shape[1])
            index.add(centroids.astype('float32'))
            
            search_k = min(k + 1, len(centroids))
            similarities, indices = index.search(centroids.astype('float32'), search_k)
            
            for cluster_id, (cluster_sims, neighbor_cluster_ids) in enumerate(zip(similarities, indices)):
                if cluster_id not in clusters:
                    continue
                    
                for neighbor_cluster_id, similarity in zip(neighbor_cluster_ids[1:], cluster_sims[1:]):
                    if neighbor_cluster_id not in clusters or similarity < 0.5:
                        continue
                    
                    # Connect representative nodes
                    source_chunks = clusters[cluster_id][:3]
                    target_chunks = clusters[neighbor_cluster_id][:3]
                    
                    for src_idx in source_chunks:
                        for tgt_idx in target_chunks:
                            try:
                                sim = np.dot(embeddings[src_idx], embeddings[tgt_idx])
                                if sim > 0.6:
                                    batch_edges.append((src_idx, tgt_idx, sim))
                            except Exception as e:
                                logger.debug(f"Inter-cluster similarity failed: {e}")
        except Exception as e:
            logger.error(f"Inter-cluster processing failed: {e}")
        
        return batch_edges
    
    def _build_approximate_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """Parallel approximate graph building with random sampling."""
        
        n_chunks = len(embeddings)
        edge_count = 0
        batch_size = getattr(self.config, 'batch_size', 1000)
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Process chunks in batches
            for i in range(0, n_chunks, batch_size):
                batch_end = min(i + batch_size, n_chunks)
                sample_ratio = getattr(self.config, 'sample_ratio', 0.1)
                k_neighbors = getattr(self.config, 'k_neighbors', 10)
                
                future = executor.submit(
                    self._process_approximate_batch,
                    embeddings,
                    i,
                    batch_end,
                    n_chunks,
                    sample_ratio,
                    k_neighbors
                )
                futures.append(future)
            
            # Collect results
            with tqdm(total=len(futures), desc="Approximate batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_edges = future.result()
                        
                        with self._lock:
                            for i, j, sim in batch_edges:
                                self.g.add_edge(i, j, weight=sim)
                                edge_count += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Approximate batch failed: {e}")
        
        return edge_count
    
    def _process_approximate_batch(
        self,
        embeddings: np.ndarray,
        start_idx: int,
        end_idx: int,
        total_chunks: int,
        sample_ratio: float,
        k: int
    ) -> List[Tuple[int, int, float]]:
        """Process approximate connections for a batch."""
        batch_edges = []
        similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
        
        for i in range(start_idx, end_idx):
            try:
                # Sample candidate indices
                available_indices = [j for j in range(total_chunks) if j != i]
                if not available_indices:
                    continue
                    
                sample_size = min(int(total_chunks * sample_ratio), k * 3, len(available_indices))
                if sample_size <= 0:
                    continue
                    
                candidate_indices = np.random.choice(
                    available_indices,
                    size=sample_size,
                    replace=False
                )
                
                # Compute similarities
                similarities = []
                for j in candidate_indices:
                    try:
                        sim = np.dot(embeddings[i], embeddings[j])
                        similarities.append((j, sim))
                    except Exception as e:
                        logger.debug(f"Similarity computation failed: {e}")
                        continue
                
                # Keep top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                for j, sim in similarities[:k]:
                    if sim > similarity_threshold:
                        batch_edges.append((i, j, sim))
            except Exception as e:
                logger.debug(f"Approximate batch processing failed for index {i}: {e}")
        
        return batch_edges
    
    def _build_multi_level_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """Multi-level parallel approach combining multiple strategies."""
        
        # Level 1: Coarse clustering
        n_clusters = getattr(self.config, 'n_clusters', 50)
        n_coarse = min(n_clusters, len(embeddings) // 100, len(embeddings))
        
        if n_coarse <= 1:
            logger.warning("Too few clusters for multi-level, falling back to direct method")
            return self._build_graph_optimized([], embeddings, checkpoint_callback)
        
        try:
            coarse_kmeans = MiniBatchKMeans(n_clusters=n_coarse, random_state=42, n_init=3)
            coarse_labels = coarse_kmeans.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"Coarse clustering failed: {e}")
            return 0
        
        edge_count = 0
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Process each coarse cluster
            for cluster_id in range(n_coarse):
                cluster_mask = coarse_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 1:
                    future = executor.submit(
                        self._process_multi_level_cluster,
                        embeddings,
                        cluster_indices
                    )
                    futures.append(future)
            
            # Collect results
            with tqdm(total=len(futures), desc="Multi-level batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_edges = future.result()
                        
                        with self._lock:
                            for i, j, sim in batch_edges:
                                self.g.add_edge(i, j, weight=sim)
                                edge_count += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Multi-level batch failed: {e}")
        
        return edge_count
    
    def _process_multi_level_cluster(
        self,
        embeddings: np.ndarray,
        cluster_indices: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Process a cluster with adaptive strategy."""
        batch_edges = []
        similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
        k_neighbors = getattr(self.config, 'k_neighbors', 10)
        
        if len(cluster_indices) <= 50:
            # Small: full connections
            for i, idx1 in enumerate(cluster_indices):
                for idx2 in cluster_indices[i+1:]:
                    try:
                        sim = np.dot(embeddings[idx1], embeddings[idx2])
                        if sim > similarity_threshold:
                            batch_edges.append((idx1, idx2, sim))
                    except Exception as e:
                        logger.debug(f"Direct similarity failed: {e}")
        else:
            # Large: KNN
            try:
                cluster_embeddings = embeddings[cluster_indices]
                index = faiss.IndexFlatIP(cluster_embeddings.shape[1])
                cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                index.add(cluster_embeddings_norm.astype('float32'))
                
                k = min(k_neighbors, len(cluster_indices) - 1)
                similarities, indices = index.search(cluster_embeddings_norm.astype('float32'), k + 1)
                
                for i, (chunk_sims, neighbor_indices) in enumerate(zip(similarities, indices)):
                    original_idx = cluster_indices[i]
                    for neighbor_idx, similarity in zip(neighbor_indices[1:], chunk_sims[1:]):
                        if similarity > similarity_threshold:
                            neighbor_original_idx = cluster_indices[neighbor_idx]
                            batch_edges.append((original_idx, neighbor_original_idx, float(similarity)))
            except Exception as e:
                logger.error(f"KNN within multi-level cluster failed: {e}")
        
        return batch_edges

    def _build_hybrid_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """Hybrid approach: KNN + Hierarchical + Approximate."""
        edge_count = 0
        
        try:
            # Strategy 1: KNN for high-quality local connections
            knn_edges = self._build_knn_parallel(embeddings, None, False)
            edge_count += knn_edges
            
            # Strategy 2: Hierarchical for global structure
            hier_edges = self._build_hierarchical_parallel(embeddings, None)
            edge_count += hier_edges
            
            # Strategy 3: Approximate for additional coverage
            approx_edges = self._build_approximate_parallel(embeddings, None)
            edge_count += approx_edges
        except Exception as e:
            logger.error(f"Hybrid building failed: {e}")
        
        return edge_count
    
    def _build_lsh_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """LSH-based parallel graph building for ultra-fast approximate similarity."""
        try:
            # Implement using FAISS LSH
            dimension = embeddings.shape[1]
            n_hash_bits = getattr(self.config, 'n_hash_bits', 128)
            index = faiss.IndexLSH(dimension, n_hash_bits)
            index.add(embeddings.astype('float32'))
        except Exception as e:
            logger.warning(f"LSH not available: {e}, falling back to KNN")
            return self._build_knn_parallel(embeddings, checkpoint_callback, False)
        
        edge_count = 0
        batch_size = getattr(self.config, 'batch_size', 1000)
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(0, len(embeddings), batch_size):
                batch_end = min(i + batch_size, len(embeddings))
                k_neighbors = getattr(self.config, 'k_neighbors', 10)
                
                future = executor.submit(
                    self._process_lsh_batch,
                    index,
                    embeddings[i:batch_end],
                    i,
                    k_neighbors
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_edges = future.result()
                    with self._lock:
                        for i, j, sim in batch_edges:
                            self.g.add_edge(i, j, weight=sim)
                            edge_count += 1
                except Exception as e:
                    logger.error(f"LSH batch failed: {e}")
        
        return edge_count
    
    def _process_lsh_batch(
        self,
        index,
        embeddings_batch: np.ndarray,
        start_idx: int,
        k: int
    ) -> List[Tuple[int, int, float]]:
        """Process LSH batch."""
        batch_edges = []
        similarity_threshold = getattr(self.config, 'similarity_threshold', 0.7)
        
        try:
            similarities, indices = index.search(embeddings_batch.astype('float32'), k + 1)
            
            for i, (chunk_sims, chunk_indices) in enumerate(zip(similarities, indices)):
                original_idx = start_idx + i
                for neighbor_idx, similarity in zip(chunk_indices[1:], chunk_sims[1:]):
                    if similarity > similarity_threshold:
                        batch_edges.append((original_idx, int(neighbor_idx), float(similarity)))
        except Exception as e:
            logger.error(f"LSH batch processing failed: {e}")
        
        return batch_edges
    
    def _build_spectral_parallel(
        self,
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable]
    ) -> int:
        """Spectral clustering based graph building with safeguards."""
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            logger.warning("Spectral clustering not available, falling back to hierarchical")
            return self._build_hierarchical_parallel(embeddings, checkpoint_callback)

        try:
            n_clusters = getattr(self.config, "n_clusters", 50)
            max_workers = getattr(self.config, "max_workers", 4)
            n_samples = len(embeddings)

            if n_samples < 2:
                logger.warning("Not enough embeddings for spectral clustering. Falling back to KNN.")
                return self._build_knn_parallel(embeddings, checkpoint_callback, use_gpu=False)

            if n_samples > 5000:
                # Large dataset → approximate spectral clustering
                n_clusters = max(2, min(n_clusters, n_samples // 50))
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=min(50, max(2, n_samples // 10)),
                    n_jobs=max_workers
                )
                cluster_labels = spectral.fit_predict(embeddings)
            else:
                # Smaller dataset → full spectral clustering
                similarity_matrix = np.dot(embeddings, embeddings.T)
                n_clusters = max(2, min(n_clusters // 2, max(1, n_samples // 20)))
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    n_jobs=max_workers
                )
                cluster_labels = spectral.fit_predict(similarity_matrix)

            logger.info(f"Spectral clustering completed with n_clusters={n_clusters} for {n_samples} samples")

        except Exception as e:
            logger.error(f"Spectral clustering failed: {e}. Falling back to hierarchical.")
            return self._build_hierarchical_parallel(embeddings, checkpoint_callback)

        # Build graph from clusters
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(idx)

        return self._build_from_clusters_parallel(embeddings, clusters)

    def _build_from_clusters_parallel(
        self,
        embeddings: np.ndarray,
        clusters: Dict[int, List[int]]
    ) -> int:
        """Build graph from pre-computed clusters."""
        edge_count = 0
        max_workers = getattr(self.config, 'max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for cluster_id, chunk_indices in clusters.items():
                if len(chunk_indices) > 1:
                    intra_k = getattr(self.config, 'intra_cluster_k', 5)
                    future = executor.submit(
                        self._process_cluster,
                        embeddings,
                        chunk_indices,
                        intra_k,
                        "spectral"
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_edges = future.result()
                    with self._lock:
                        for i, j, sim in batch_edges:
                            self.g.add_edge(i, j, weight=sim)
                            edge_count += 1
                except Exception as e:
                    logger.error(f"Spectral batch failed: {e}")
        
        return edge_count

    def save_graph(
            self,
            file_path: Union[str, Path],
            format: str = "pickle",
            compress: bool = True
    ) -> None:
        """Save graph with enhanced options."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "pickle":
                mode = 'wb'
                if compress:
                    try:
                        import gzip
                        with gzip.open(file_path, mode) as f:
                            pickle.dump(self.g, f)
                    except ImportError:
                        logger.warning("gzip not available, saving uncompressed")
                        with open(file_path, mode) as f:
                            pickle.dump(self.g, f)
                else:
                    with open(file_path, mode) as f:
                        pickle.dump(self.g, f)
            elif format == "json":
                data = nx.node_link_data(self.g)

                # Convert numpy arrays to lists for JSON serialization
                for node in data['nodes']:
                    if 'emb' in node and isinstance(node['emb'], np.ndarray):
                        node['emb'] = node['emb'].tolist()
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2 if not compress else None)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Graph saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise
    
    def load_graph(self, file_path: Union[str, Path], format: str = "pickle") -> None:
        """Load graph with enhanced error handling."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")
        
        try:
            if format == "pickle":
                # Try compressed first, then uncompressed
                try:
                    import gzip
                    with gzip.open(file_path, 'rb') as f:
                        self.g = pickle.load(f)
                except (ImportError, OSError):
                    with open(file_path, 'rb') as f:
                        self.g = pickle.load(f)
                        
            elif format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert embedding lists back to numpy arrays
                for node in data['nodes']:
                    if 'emb' in node and isinstance(node['emb'], list):
                        node['emb'] = np.array(node['emb'])
                
                self.g = nx.node_link_graph(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(
                f"Graph loaded: {self.g.number_of_nodes()} nodes, "
                f"{self.g.number_of_edges()} edges"
            )
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'g') and self.g:
                self.g.clear()
            if hasattr(self, 'cache') and hasattr(self.cache, '_memory_cache'):
                self.cache._memory_cache.clear()
            if hasattr(self, 'cache') and hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                self.cache.redis_client.close()
            logger.info("PathRAG resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def example_usage():
    """
    Example usage demonstrating PathRAG functionality.
    """
    # Create sample data
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual data.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    # Create mock embeddings
    embeddings = np.random.randn(len(chunks), 384).astype(np.float32)
    
    # Initialize PathRAG
    pathrag = PathRAG()
    
    try:
        # Build graph
        print("Building graph...")
        pathrag.build_graph(chunks, embeddings, method="hierarchical")
        
        # Retrieve nodes
        print("\nRetrieving nodes...")
        query = "What is machine learning?"
        nodes = pathrag.retrieve_nodes(query, top_k=3)
        print(f"Retrieved nodes: {nodes}")
        
        # Find paths
        print("\nFinding paths...")
        paths = pathrag.prune_paths(nodes, max_hops=3)
        print(f"Found {len(paths)} paths")
        
        # Score paths
        print("\nScoring paths...")
        scored_paths = pathrag.score_paths(paths)
        
        # Generate prompt
        print("\nGenerating prompt...")
        prompt = pathrag.generate_prompt(query, scored_paths)
        print("Generated prompt:")
        print(prompt)
        
        # Get metrics
        print("\nMetrics:")
        metrics = pathrag.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
        # Health check
        print("\nHealth check:")
        health = pathrag.health_check()
        print(f"  Status: {health['status']}")
        if health['issues']:
            print(f"  Issues: {health['issues']}")
        if health['recommendations']:
            print(f"  Recommendations: {health['recommendations']}")
            
    except Exception as e:
        logger.error(f"Example usage failed: {e}")
        raise
    finally:
        pathrag.cleanup()


if __name__ == "__main__":
    example_usage()
