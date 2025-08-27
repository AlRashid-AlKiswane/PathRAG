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
from time import time
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Union
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging, MemoryMonitor
from src.schemas import PathRAGConfig
from src.utils import timer
from .graph_cache import GraphCache
from .path_rag_metrics import PathRAGMetrics
from src.helpers import get_settings, Settings
from src.llms_providers import HuggingFaceModel

logger = setup_logging(name="PATH-RAG")
app_settings: Settings = get_settings()


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
        embedding_model: HuggingFaceModel,
        config: Optional[PathRAGConfig] = None
    ):
        """
        Initialize PathRAG with production-ready configuration.
        
        Args:
            embedding_model: Model for text embedding
            config: Configuration object with parameters
        """
        self.config = config or PathRAGConfig()
        self.embedding_model: HuggingFaceModel = embedding_model
        self.g = nx.DiGraph()
        self.metrics = PathRAGMetrics()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        self.cache = GraphCache(self.config)
        self._lock = threading.Lock()
        
        # Setup logging        
        logger.info(f"PathRAG initialized with config: {self.config}")
    
    @timer('graph_build_time')
    def build_graph(
        self, 
        chunks: List[str], 
        embeddings: np.ndarray,
        checkpoint_callback: Optional[callable] = None
    ) -> None:
        """
        Build semantic graph with production optimizations.
        
        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings
            checkpoint_callback: Optional callback for progress checkpoints
        """
        if not self._validate_inputs(chunks, embeddings):
            raise ValueError("Invalid inputs provided")
        
        if len(chunks) > self.config.max_graph_size:
            logger.warning(
                f"Input size {len(chunks)} exceeds max_graph_size"
                f"{self.config.max_graph_size}, truncating"
            )
            chunks = chunks[:self.config.max_graph_size]
            embeddings = embeddings[:self.config.max_graph_size]
        
        with self.memory_monitor.memory_guard():
            self._build_graph_optimized(chunks, embeddings, checkpoint_callback)

    def _validate_inputs(self, chunks: List[str], embeddings: np.ndarray) -> bool:
        """Comprehensive input validation."""
        if not chunks or not isinstance(chunks, list):
            logger.error("Invalid chunks provided")
            return False

        if not isinstance(embeddings, np.ndarray):
            logger.error("Embeddings must be numpy array")
        
        if len(chunks) != embeddings.shape[0]:
            logger.error("Chunks and embeddings length mismatch")
            return False
        
        if embeddings.ndim != 2:
            logger.error("Embeddings must be 2D array")
            return False
        
        return True

    def _build_graph_optimized(
            self,
            chunks: List[str],
            embeddings: np.ndarray,
            checkpoint_callback: Optional[callable]
    ) -> None:
        """Optimized graph building with batching and concurrency."""
        self.g.clear()
        logger.info("Building graph for %d chunks", int(len(chunks)))

        # Add nodes efficiently
        with self._lock:
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self.g.add_node(idx, text=chunk, emb=emb)
        
        # Build adges in batches with concurrent processing
        total_pairs = len(chunks) * (len(chunks) -1) // 2
        processed_pairs = 0
        edge_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            # Submit batch jobs
            for i in range(0, len(chunks), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(chunks))
                future = executor.submit(
                    self._process_batch,
                    embeddings,
                    i,
                    batch_end,
                    len(chunks)
                )
                futures.append(future)

            # Collect results wiht progress tracking
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
                        if checkpoint_callback and processed_pairs % self.config.checkpoint_interval == 0:
                            checkpoint_callback(processed_pairs, total_pairs)

                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")

        self.metrics.set_metric('nodes_processed', len(chunks))
        self.metrics.set_metric('edges_created', edge_count)
        logger.info(f"Graph built: {len(chunks)} nodes, {edge_count} edges")
    
    def _process_batch(
        self,
        embeddings: np.ndarray,
        start_idx: int,
        end_idx: int,
        total_size: int
    ) -> List[Tuple[int, int, float]]:
        """Process a batch of similarity computations."""
        batch_edges = []

        for i in tqdm(range(start_idx, end_idx)):
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
                    
                    if sim > self.config.sim_threshold:
                        batch_edges.append((i, j, float(sim)))
                
                except Exception as e:
                    logger.debug(f"Similarity computation failed for ({i}, {j}): {e}")
                    continue
        
        return batch_edges
    
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
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics.increment("cache_hist")
                return cached_result
            self.metrics.increment("cache_hist")
        
        try:
            # Embed quyer with error handling
            q_emb = self._embed_query_safe(query)

            # compute similarites efficiently
            similarities = self._compute_similarities_batch(q_emb)

            # Get top-k results
            top_nodes = heapq.nlargest(
                top_k,
                similarities.items(),
                key=lambda x: x[1]
            )
            result = [node_id for node_id, _ in top_nodes]

            # Cache result
            if cache_key:
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
        if self.g.number_of_nodes() == 0:
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
                time.sleep(0.1 * (attempt + 1)) 
    
    @lru_cache(maxsize=1000)
    def _compute_similarities_batch(self, q_emb_tuple: tuple) -> Dict[int, float]:
        """Compute similarities with caching (using tuple for hashability)."""
        q_emb = np.array(q_emb_tuple)
        similarities = {}

        # Batch process similarites
        node_ids = list(self.g.nodes())
        embedding_martix = np.vstack([
            self.g.nodes[ndi]['emb'] for ndi in node_ids
        ])

        # Vectorized similarity computation
        sims = cosine_similarity(q_emb.reshape(1, -1), embedding_martix)[0]

        for node_id, sim in zip(node_ids, sims):
            similarities[node_id] = float(sim)
        
        return similarities
    

    @timer('path_pruning_time')
    def prune_paths(
        self,
        nodes: List[int],
        max_hops: int = 4,
        max_paths_per_pair: int = 10
    ) -> List[List[int]]:
        """
        Prune paths with production optimizations.
        
        Args:
            nodes: List of node indices
            max_hops: Maximum path length
            max_paths_per_pair: Limit paths per node pair for performance
            
        Returns:
            List of valid paths
        """
        if not self._validate_path_inputs(nodes, max_hops):
            raise ValueError("Invalid path pruning parameters")

        valid_paths = []
        total_pairs = len(nodes) * (len(nodes) - 1) // 2

        logger.info(f"Starting path pruning for {len(nodes)} nodes")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []

            # Submit path finding jobs
            for i, u in enumerate(nodes):
                for v in nodes[i +1:]:
                    future = executor.submit(
                        self._fin
                    )
    def _validate_path_inputs(self, nodes: List[int], max_hops: int) -> bool:
        """Validate path pruning inputs."""
        if not nodes or not isinstance(nodes, list):
            return False
        if max_hops < 1 or not isinstance(max_hops, int):
            return False
        if not all(isinstance(n, int) and self.g.has_node(n) for n in nodes):
            return False
        return True
    
    def _find_paths_between_nodes(
            self,
            u: int,
            v: int,
            max_hops: int,
            max_paths: int
    ) -> List[List[int]]:
        """Find valid paths between tow nodes."""
        valid_paths = []

        try:
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
                if score >= self.config.prune_thresh:
                    valid_paths.append(path)

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
            for i in range(len(path) -1):
                edge_data = self.g.edges.get((path[i], path[i + 1]))
                if edge_data is None:
                    return 0.0
                weights.append(edge_data['weight'])
            base_core = np.prod(weights)
            decay_factor = self.config.decay_rate ** (len(path) - 1)

            return float(base_core * decay_factor)
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
                text = self.g.nodes[node_id].get("text", "")
                preview = text[:max_length] + "..." if len(text) > max_length else text
                texts.append(preview)
            return " → ".join(texts)

        except Exception:
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
                node_data = self.g.nodes.get(node_id, {})
                text = node_data.get("text", f"Node_{node_id}")

                # Truncate long texts
                if len(text) > 100:
                    text = text[:97] + "..."
                path_texts.append(text)

            return f"{index}. [Score: {score:.3f}] {' → '.join(path_texts)}"
            
        except Exception as e:
            logger.debug(f"Path formatting failed: {e}")
            return f"{index}. [Error formatting path]"
    
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
                    import gzip
                    with gzip.open(file_path, mode) as f:
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
                except:
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        base_metrics = self.metrics.get_report()
        
        # Add graph statistics
        graph_stats = {
            'nodes_count': self.g.number_of_nodes(),
            'edges_count': self.g.number_of_edges(),
            'memory_usage_gb': self.memory_monitor.get_memory_usage(),
            'avg_degree': np.mean([d for n, d in self.g.degree()]) if self.g.nodes() else 0
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
        if memory_usage > self.config.memory_limit_gb * 0.8:
            health['issues'].append(f"High memory usage: {memory_usage:.2f}GB")
            health['recommendations'].append("Consider reducing batch size or graph size")
        
        # Check graph size
        if self.g.number_of_nodes() > self.config.max_graph_size * 0.9:
            health['issues'].append("Graph approaching size limit")
            health['recommendations'].append("Consider graph pruning or increasing limits")
        
        # Set overall status
        if health['issues']:
            health['status'] = 'warning' if len(health['issues']) < 3 else 'critical'
        
        return health
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.g.clear()
            self.cache._memory_cache.clear()
            if self.cache.redis_client:
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
