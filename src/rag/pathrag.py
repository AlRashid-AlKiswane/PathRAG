"""
Module implementing PathRAG (Path-aware Retrieval-Augmented Generation), a
framework that enhances retrieval and reasoning in LLM systems using graph-based
relational paths among document chunks.

This module includes the PathRAG class which performs the following:
- Constructs a semantic similarity graph from text chunks.
- Retrieves relevant subgraphs based on a query.
- Prunes paths using decay-based flow scoring.
- Generates prompts incorporating evidence paths.

Dependencies:
- networkx
- numpy
- sentence-transformers
- scikit-learn
- src.utils (for logging setup)

Typical usage:
    rag = PathRAG()
    rag.build_graph(chunks)
    top_nodes = rag.retrieve_nodes("What is AI?", top_k=5)
    paths = rag.prune_paths(top_nodes)
    scored = rag.score_paths(paths)
    prompt = rag.generate_prompt("What is AI?", scored)
"""
import os
import sys
import logging
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

from src.utils import setup_logging

logger = setup_logging()

class PathRAG:
    """
    PathRAG: Embedding-enhanced, graph-based RAG with path pruning and relational path prompting.

    Attributes:
        g (nx.DiGraph): Directed graph storing text chunks and their relationships.
        decay_rate (float): Weight decay per edge hop for path scoring.
        prune_thresh (float): Minimum flow score to keep a path during pruning.
        embed_model (SentenceTransformer): Sentence embedding model.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        decay_rate: float = 0.8,
        prune_thresh: float = 0.01,
        sim_should: float = 0.1
    ) -> None:
        """
        Initialize PathRAG with embedding model and path parameters.

        Args:
            embedding_model: Name of HuggingFace sentence embedding model.
            decay_rate: Weight decay factor applied per edge hop in paths.
            prune_thresh: Minimum flow score threshold for path pruning.

        Raises:
            ValueError: If invalid parameters are provided.
            RuntimeError: If embedding model fails to load.
        """
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 and 1")
        if prune_thresh < 0:
            raise ValueError("prune_thresh must be non-negative")

        try:
            self.embed_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model loading failed: {e}")

        self.g = nx.DiGraph()
        self.decay_rate = decay_rate
        self.prune_thresh = prune_thresh
        self.sim_should = sim_should
        logger.info("PathRAG initialized with decay_rate=%.2f, prune_thresh=%.3f",
                   decay_rate, prune_thresh)

    def build_graph(self, chunks: List[str]) -> None:
        """
        Build graph with chunk nodes and semantic edge weights.

        Args:
            chunks: List of pre-segmented text chunks.

        Raises:
            ValueError: If empty chunk list is provided.
            RuntimeError: If embedding generation fails.
        """
        if not chunks:
            raise ValueError("No text chunks provided")
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of strings")

        self.g.clear()
        embeddings = []
        logger.info("Embedding %d chunks", len(chunks))

        for idx, text in enumerate(chunks):
            if not isinstance(text, str) or not text.strip():
                logger.warning("Empty or non-string chunk at index %d", idx)
                continue

            try:
                emb = self.embed_model.encode(text, convert_to_numpy=True)
                if emb.ndim != 1:
                    raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")
            except Exception as e:
                logger.error("Embedding failed for chunk %d: %s", idx, str(e))
                raise RuntimeError(f"Embedding generation failed: {e}")

            embeddings.append(emb)
            self.g.add_node(idx, text=text, emb=emb)
            logger.debug("Added node %d with %d-dim embedding", idx, len(emb))

        # Add weighted edges based on semantic similarity
        edge_count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                try:
                    sim = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0, 0]
                    sim = float(sim)
                except Exception as e:
                    logger.warning("Similarity computation failed for nodes (%d,%d): %s", i, j, e)
                    continue

                if sim > self.sim_should:
                    self.g.add_edge(i, j, weight=sim)
                    self.g.add_edge(j, i, weight=sim)  # Undirected behavior
                    edge_count += 2
                    logger.debug("Added edge (%d,%d) with weight %.3f", i, j, sim)

        logger.info(
            "Graph built with %d nodes and %d edges",
            self.g.number_of_nodes(),
            self.g.number_of_edges()
        )

    def retrieve_nodes(self, query: str, top_k: int = 5) -> List[int]:
        """
        Retrieve top_k semantically similar nodes to the query.

        Args:
            query: Input question string.
            top_k: Number of most similar nodes to return.

        Returns:
            List of node indices sorted by descending similarity.

        Raises:
            ValueError: If query is empty or top_k is invalid.
            RuntimeError: If query embedding fails.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive integer")
        if self.g.number_of_nodes() == 0:
            raise RuntimeError("Graph is empty - build_graph() first")

        try:
            q_emb = self.embed_model.encode(query, convert_to_numpy=True)
            if q_emb.ndim != 1:
                raise RuntimeError(f"Unexpected query embedding shape: {q_emb.shape}")
        except Exception as e:
            logger.error("Query embedding failed: %s", str(e))
            raise RuntimeError(f"Query embedding failed: {e}")

        sims = {}
        for nid, data in self.g.nodes(data=True):
            try:
                node_emb = data["emb"]
                similarity = cosine_similarity(
                    q_emb.reshape(1, -1),
                    node_emb.reshape(1, -1)
                )[0, 0]
                sims[nid] = float(similarity)
            except KeyError:
                logger.warning("Node %d missing embedding", nid)
                continue
            except Exception as e:
                logger.warning("Similarity computation failed for node %d: %s", nid, e)
                continue

        if not sims:
            raise RuntimeError("No valid node similarities computed")

        ranked = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:top_k]
        logger.info(
            "Retrieved top %d nodes: %s with similarities %s",
            top_k,
            ranked,
            [sims[n] for n in ranked]
        )
        return ranked

    def prune_paths(self, nodes: List[int], max_hops: int = 4) -> List[List[int]]:
        """
        Prune relational paths based on decay flow pruning.

        Args:
            nodes: Selected node IDs to consider for paths.
            max_hops: Maximum number of hops in any path.

        Returns:
            List of filtered node paths that meet the threshold.

        Raises:
            ValueError: If invalid nodes or parameters provided.
        """
        if not nodes:
            raise ValueError("Empty node list provided")
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        if any(n not in self.g for n in nodes):
            raise ValueError("One or more nodes not found in graph")

        paths = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                try:
                    for path in nx.all_simple_paths(
                        self.g, source=u, target=v, cutoff=max_hops
                    ):
                        flow = self._compute_flow(path)
                        if flow >= self.prune_thresh:
                            paths.append(path)
                            logger.debug(
                                "Kept path %s with flow %.3f",
                                path,
                                flow
                            )
                except nx.NetworkXError as e:
                    logger.warning("Path finding failed for (%d,%d): %s", u, v, e)
                    continue

        logger.info(
            "Path pruning retained %d paths from %d source nodes",
            len(paths),
            len(nodes)
        )
        return paths

    def _compute_flow(self, path: List[int]) -> float:
        """
        Compute flow score: product of edge weights with decay^hops.

        Args:
            path: List of node IDs representing the path.

        Returns:
            Flow score as float.

        Raises:
            ValueError: If path is too short or invalid.
        """
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        # Calculate base flow as product of edge weights
        try:
            edge_weights = [
                self.g.edges[path[i], path[i + 1]]["weight"]
                for i in range(len(path) - 1)
            ]
            base_flow = np.prod(edge_weights)
        except KeyError as e:
            logger.warning("Missing edge weight in path %s: %s", path, e)
            return 0.0

        # Apply decay based on path length
        return base_flow * (self.decay_rate ** (len(path) - 1))

    def score_paths(self, paths: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Score each path and sort ascending (for prompting).

        Args:
            paths: List of node ID sequences.

        Returns:
            List of dictionaries with 'path' and 'score' keys, sorted by score.

        Raises:
            ValueError: If empty paths list provided.
        """
        if not paths:
            raise ValueError("Empty paths list provided")

        scored = []
        for p in paths:
            try:
                score = self._compute_flow(p)
                scored.append({"path": p, "score": score})
            except ValueError as e:
                logger.warning("Skipping invalid path %s: %s", p, e)
                continue

        scored.sort(key=lambda x: x["score"])
        logger.debug(
            "Path scoring complete. Top path: %s with score %.3f",
            scored[-1]["path"] if scored else None,
            scored[-1]["score"] if scored else 0.0
        )
        return scored

    def generate_prompt(self, query: str, scored_paths: List[Dict[str, Any]]) -> str:
        """
        Formulate the concatenated prompt with paths and query.

        Args:
            query: User question string.
            scored_paths: Output of score_paths().

        Returns:
            Fully formatted prompt string.

        Raises:
            ValueError: If query is empty or paths invalid.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if not isinstance(scored_paths, list):
            raise ValueError("scored_paths must be a list")

        lines = ["QUERY: " + query, "RELATED EVIDENCE PATHS:"]

        for item in scored_paths:
            if not isinstance(item, dict) or "path" not in item:
                logger.warning("Invalid path item skipped: %s", item)
                continue

            try:
                nodes = item["path"]
                txts = []
                for n in nodes:
                    if n not in self.g:
                        raise ValueError(f"Node {n} not found in graph")
                    txts.append(self.g.nodes[n]["text"])
                path_str = " → ".join(txts)
                lines.append(f"- [Score: {item['score']:.3f}] {path_str}")
            except Exception as e:
                logger.warning("Failed to process path %s: %s", item["path"], e)
                continue

        prompt = "\n".join(lines)
        logger.info(
            "Generated prompt with %d paths (total length: %d chars)",
            len(scored_paths),
            len(prompt)
        )
        return prompt
