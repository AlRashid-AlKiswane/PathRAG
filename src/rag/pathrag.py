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

import heapq
import logging
import os
import sys
from typing import Any, Dict, List

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

from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.llms_providers import HuggingFaceModel

logger = setup_logging(name="PATH-RAG")
app_settings: Settings = get_settings()


class PathRAG:
    """
    Path-aware Retrieval-Augmented Generation (PathRAG) class implementing graph-based
    relational reasoning for information retrieval and prompt construction.

    This class builds a semantic directed graph from embedded text chunks, computes
    similarity edges between nodes, and supports retrieving relevant nodes and relational
    paths between them. Paths are scored with decay-weighted flow to prune and prioritize
    the most relevant chains of evidence.

    Attributes:
        g (nx.DiGraph): Directed graph storing nodes as chunk indices and edges weighted by cosine similarity.
        decay_rate (float): Multiplicative decay applied per hop in path scoring (0 < decay_rate ≤ 1).
        prune_thresh (float): Minimum flow score threshold below which paths are discarded.
        sim_should (float): Similarity threshold for creating edges between nodes (0 ≤ sim_should ≤ 1).
        embedding_model (HuggingFaceModel): Model instance used to embed query texts.
    """

    def __init__(
        self,
        embedding_model: HuggingFaceModel,
        decay_rate: float = 0.8,
        prune_thresh: float = 0.01,
        sim_should: float = 0.1
    ) -> None:
        """
        Initialize PathRAG with specified parameters and an embedding model.

        Args:
            embedding_model (HuggingFaceModel): Pretrained model to generate vector embeddings from text.
            decay_rate (float, optional): Decay factor per hop in path scoring; must be in (0, 1]. Defaults to 0.8.
            prune_thresh (float, optional): Minimum path flow score to retain path; must be non-negative. Defaults to 0.01.
            sim_should (float, optional): Cosine similarity threshold to create edges between nodes. Defaults to 0.1.

        Raises:
            ValueError: If decay_rate is not in (0,1] or prune_thresh is negative.
        """
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 (exclusive) and 1 (inclusive)")
        if prune_thresh < 0:
            raise ValueError("prune_thresh must be non-negative")

        self.g = nx.DiGraph()
        self.decay_rate = decay_rate
        self.prune_thresh = prune_thresh
        self.sim_should = sim_should
        self.embedding_model: HuggingFaceModel = embedding_model

        logger.info(
            "Initialized PathRAG (decay=%.2f, prune_thresh=%.3f, sim_thresh=%.3f)",
            decay_rate, prune_thresh, sim_should
        )

    def build_graph(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Construct a semantic graph where nodes represent text chunks and edges represent
        cosine similarity above a threshold between chunk embeddings.

        Nodes store the chunk text and embedding. Edges are bidirectional with weights
        equal to cosine similarity between node embeddings.

        Args:
            chunks (List[str]): List of text chunks to include as graph nodes.
            embeddings (np.ndarray): 2D numpy array where each row is the embedding vector for a chunk.

        Raises:
            ValueError: If chunks is empty, not a list, or if embeddings is not a numpy array,
                        or if the number of chunks does not match number of embeddings.
            RuntimeError: If an unexpected error occurs during graph construction.
        """
        if not chunks or not isinstance(chunks, list):
            raise ValueError("Invalid or empty chunk list provided.")
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy ndarray.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings size mismatch.")

        self.g.clear()
        logger.info("Building semantic graph for %d chunks...", len(chunks))

        try:
            # Add nodes with associated chunk text and embeddings
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self.g.add_node(idx, text=chunk, emb=emb)

            edge_count = 0
            # Add edges for pairs with similarity above threshold
            for i in tqdm(range(len(embeddings)), desc="Building edges"):
                for j in range(i + 1, len(embeddings)):
                    try:
                        sim = cosine_similarity(
                            embeddings[i].reshape(1, -1),
                            embeddings[j].reshape(1, -1)
                        )[0, 0]
                        if sim > self.sim_should:
                            self.g.add_edge(i, j, weight=sim)
                            self.g.add_edge(j, i, weight=sim)
                            edge_count += 2
                    except Exception as e:
                        logger.warning("Similarity calculation failed for nodes (%d, %d): %s", i, j, e)

            logger.info("Graph constructed with %d nodes and %d edges", self.g.number_of_nodes(), edge_count)
        except Exception as e:
            logger.error("Graph construction failed: %s", e)
            raise RuntimeError("Failed to build semantic graph.") from e

    def retrieve_nodes(self, query: str, top_k: int = 5) -> List[int]:
        """
        Retrieve indices of the top_k graph nodes most semantically similar to the query.

        The query text is embedded and cosine similarity is computed with each node embedding.

        Args:
            query (str): Text query string to retrieve relevant nodes for.
            top_k (int, optional): Number of top similar nodes to return. Must be > 0. Defaults to 5.

        Returns:
            List[int]: List of node indices ranked by descending similarity to the query.

        Raises:
            ValueError: If query is empty or not a string, or if top_k ≤ 0.
            RuntimeError: If the graph is empty or query embedding fails, or no similarities computed.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.g.number_of_nodes() == 0:
            raise RuntimeError("Graph is empty - please call build_graph() before retrieval.")

        try:
            q_emb = self.embedding_model.embed_texts(texts=query, convert_to_numpy=True)
            if isinstance(q_emb, list):
                q_emb = np.array(q_emb)
            if q_emb.ndim == 2 and q_emb.shape[0] == 1:
                q_emb = q_emb[0]
            if q_emb.ndim != 1:
                raise RuntimeError(f"Unexpected query embedding shape: {q_emb.shape}")
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            raise RuntimeError(f"Query embedding failed: {e}") from e

        sims = {}
        for nid, data in self.g.nodes(data=True):
            try:
                similarity = cosine_similarity(
                    q_emb.reshape(1, -1),
                    data["emb"].reshape(1, -1)
                )[0, 0]
                sims[nid] = float(similarity)
            except Exception as e:
                logger.warning("Similarity computation failed for node %d: %s", nid, e)

        if not sims:
            raise RuntimeError("No valid node similarities computed.")

        ranked = heapq.nlargest(top_k, sims.items(), key=lambda x: x[1])
        ranked_nodes = [nid for nid, _ in ranked]
        logger.info("Top %d retrieved nodes: %s", top_k, ranked_nodes)
        return ranked_nodes

    def prune_paths(self, nodes: List[int], max_hops: int = 4) -> List[List[int]]:
        """
        Find and prune relational paths between specified nodes using decay-weighted scoring.

        Enumerates all simple paths between each pair of nodes up to max_hops length.
        Each path's flow score is computed and only those above prune_thresh are retained.

        Args:
            nodes (List[int]): List of node indices to consider as start/end points for paths.
            max_hops (int, optional): Maximum path length (number of edges) to consider. Must be ≥ 1. Defaults to 4.

        Returns:
            List[List[int]]: List of valid paths (each a list of node indices) passing the pruning threshold.

        Raises:
            ValueError: If nodes is empty or not a list, or if max_hops < 1.
        """
        if not nodes or not isinstance(nodes, list):
            raise ValueError("Invalid node list.")
        if max_hops < 1:
            raise ValueError("max_hops must be ≥ 1.")

        valid_paths = []
        logger.info("Starting path pruning over %d nodes with max_hops=%d...", len(nodes), max_hops)

        total_pairs = len(nodes) * (len(nodes) - 1) // 2
        with tqdm(total=total_pairs, desc="Path pruning - node pairs", unit="pairs") as pbar:
            for i, u in enumerate(nodes):
                for v in nodes[i + 1:]:
                    try:
                        all_paths = nx.all_simple_paths(self.g, source=u, target=v, cutoff=max_hops)
                        for path in all_paths:
                            score = self._compute_flow(path)
                            if score >= self.prune_thresh:
                                valid_paths.append(path)
                        pbar.update(1)
                    except nx.NetworkXNoPath:
                        pbar.update(1)
                    except Exception as e:
                        logger.warning("Path search failed between %d and %d: %s", u, v, e)

        logger.info("Pruned to %d valid paths.", len(valid_paths))
        return valid_paths

    def _compute_flow(self, path: List[int]) -> float:
        """
        Compute the decay-weighted flow score for a given path in the graph.

        The flow score is the product of all edge weights along the path,
        multiplied by decay_rate raised to the path length minus one.

        Args:
            path (List[int]): List of node indices representing a path.

        Returns:
            float: The computed flow score. Zero if path length < 2 or on failure.
        """
        if len(path) < 2:
            return 0.0
        try:
            weights = [
                self.g.edges[path[i], path[i + 1]]["weight"]
                for i in range(len(path) - 1)
            ]
            base = np.prod(weights)
            flow = base * (self.decay_rate ** (len(path) - 1))

            return flow
        except Exception as e:
            logger.warning("Failed to compute flow for path %s: %s", path, e)
            return 0.0

    def score_paths(self, paths: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Score and sort paths based on their decay-weighted flow score.

        Paths with zero or negative scores are excluded from the results.

        Args:
            paths (List[List[int]]): List of paths, each a list of node indices.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with keys 'path' and 'score',
            sorted descending by score.

        Raises:
            ValueError: If paths is empty or None.
        """
        if not paths:
            raise ValueError("No paths to score.")

        results = []
        for path in paths:
            score = self._compute_flow(path)
            if score > 0:
                results.append({"path": path, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def generate_prompt(self, query: str, scored_paths: List[Dict[str, Any]]) -> str:
        """
        Generate a text prompt consisting of the query followed by
        a list of related evidence paths with their flow scores.

        Args:
            query (str): Original user query string.
            scored_paths (List[Dict[str, Any]]): List of scored paths with keys 'path' and 'score'.

        Returns:
            str: Formatted prompt string including query and evidence paths.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        lines = [f"QUERY: {query}", "RELATED EVIDENCE PATHS:"]
        for item in scored_paths:
            try:
                path_texts = [self.g.nodes[n]["text"] for n in item["path"]]
                line = f"- [Score: {item['score']:.3f}] " + " → ".join(path_texts)
                lines.append(line)
            except Exception as e:
                logger.warning("Failed to format path %s: %s", item.get("path", []), e)

        prompt = "\n".join(lines)
        logger.info("Prompt generated with %d paths (%d characters)", len(scored_paths), len(prompt))
        return prompt
    
    def save_graph(self, file_path: Union[str, Path], format: str = "pickle") -> None:
        """
        Save the graph to disk in the specified format. If file exists, merges
        new data without duplicating nodes or edges.

        Args:
            file_path: Path to save the graph.
            format: "pickle" or "json".

        Raises:
            ValueError: For unsupported formats.
            IOError: On save failure.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_path.exists():
                # Load existing graph
                if format == "pickle":
                    with open(file_path, 'rb') as f:
                        existing_graph = pickle.load(f)
                elif format == "json":
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                    existing_graph = nx.node_link_graph(existing_data)
                else:
                    raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")

                # Merge without duplication
                for node, attrs in self.g.nodes(data=True):
                    if not existing_graph.has_node(node):
                        existing_graph.add_node(node, **attrs)

                for u, v, attrs in self.g.edges(data=True):
                    if not existing_graph.has_edge(u, v):
                        existing_graph.add_edge(u, v, **attrs)

                # Save back
                if format == "pickle":
                    with open(file_path, 'wb') as f:
                        pickle.dump(existing_graph, f)
                elif format == "json":
                    with open(file_path, 'w') as f:
                        json.dump(nx.node_link_data(existing_graph), f)

            else:
                # First save — just write the graph
                if format == "pickle":
                    with open(file_path, 'wb') as f:
                        pickle.dump(self.g, f)
                elif format == "json":
                    with open(file_path, 'w') as f:
                        json.dump(nx.node_link_data(self.g), f)
                else:
                    raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")

            logger.info(f"Graph saved to {file_path} in {format} format")

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise

    def load_graph(self, file_path: Union[str, Path], format: str = "pickle") -> None:
        """
        Load graph from disk.
        
        Args:
            file_path: Path to load graph from
            format: Format of the saved file ("pickle" or "json")
            
        Raises:
            ValueError: If invalid format specified
            IOError: If load operation fails
        """
        if format == "pickle":
            with open(file_path, 'rb') as f:
                self.g = pickle.load(f)
        elif format == "json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.g = nx.node_link_graph(data)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")
            
        logger.info(f"Graph loaded from {file_path} ({self.g.number_of_nodes()} nodes, "
                   f"{self.g.number_of_edges()} edges)")

    def visualize_graph(self, max_nodes: int = 100) -> go.Figure:
        """
        Generate a 3D interactive visualization of the semantic graph.

        Nodes represent document chunks or concepts. Edges represent semantic similarity.
        Node color indicates degree; node type (chunk vs concept) can be differentiated.

        Args:
            max_nodes (int): Maximum number of nodes to visualize (for performance).
        
        Returns:
            plotly.graph_objects.Figure: A 3D Plotly figure.
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Graph is empty - nothing to visualize")

        # Reduce size for performance if needed
        g = self.g
        if g.number_of_nodes() > max_nodes:
            sampled_nodes = list(g.nodes())[:max_nodes]
            g = g.subgraph(sampled_nodes).copy()

        # Layout in 3D
        pos = nx.spring_layout(g, dim=3, seed=42)

        # Edge trace
        edge_x, edge_y, edge_z = [], [], []
        for src, tgt in g.edges():
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        )

        # Node trace
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_color = []
        node_size = []

        degrees = dict(g.degree())

        for node in g.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            # Chunk text preview
            node_data = g.nodes[node]
            label = node_data.get("label", "chunk")
            text = node_data.get("text", "")[:50] + "..." if "text" in node_data else label
            degree = degrees.get(node, 1)

            node_text.append(f"Node: {node}<br>Degree: {degree}<br>Type: {label}<br>{text}")

            # Node color by type
            if label == "chunk":
                node_color.append("blue")
            elif label == "concept":
                node_color.append("green")
            elif label == "query":
                node_color.append("red")
            else:
                node_color.append("gray")

            # Node size by degree
            node_size.append(5 + 10 * (degree / max(degrees.values())))

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hoverinfo='text',
            text=node_text
        )

        # Build figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="PathRAG Semantic Graph",
                margin=dict(l=0, r=0, t=50, b=0),
                scene=dict(
                    xaxis=dict(showbackground=False),
                    yaxis=dict(showbackground=False),
                    zaxis=dict(showbackground=False)
                ),
                showlegend=False,
                hovermode='closest'
            )
        )

        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the graph to a pandas DataFrame representation.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['source', 'target', 'weight', 'source_text', 'target_text']
        """
        rows = []
        for u, v, data in self.g.edges(data=True):
            rows.append({
                'source': u,
                'target': v,
                'weight': data['weight'],
                'source_text': self.g.nodes[u]['text'][:100],  # Truncate for display
                'target_text': self.g.nodes[v]['text'][:100]
            })
            
        return pd.DataFrame(rows)
