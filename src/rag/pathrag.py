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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import logging
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlite3 import Connection
from itertools import islice
from tqdm import tqdm

# Setup project basepath
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

from src.infra import setup_logging
from src.db import pull_from_table
from src.llms_providers import HuggingFaceModel
logger = setup_logging()




class PathRAG:
    """
    PathRAG: Graph-based Retrieval-Augmented Generation system that builds a 
    semantic graph of text chunks and enables path-based reasoning with pruning.

    Attributes:
        g (nx.DiGraph): Directed graph storing chunk nodes and semantic edges.
        decay_rate (float): Decay factor applied per edge hop for flow scoring.
        prune_thresh (float): Minimum flow score to retain a path.
        embed_model (SentenceTransformer): Sentence embedding model instance.
        sim_should (float): Similarity threshold for connecting edges.
        conn (Connection): SQLite database connection for retrieving embeddings.
    """

    def __init__(
        self,
        conn: Connection,
        embedding_model: HuggingFaceModel,
        decay_rate: float = 0.8,
        prune_thresh: float = 0.01,
        sim_should: float = 0.1
    ) -> None:
        """
        Initialize PathRAG with database connection, embedding model, and graph parameters.

        Args:
            conn (Connection): SQLite connection.
            embedding_model (str): HuggingFace-compatible embedding model name.
            decay_rate (float): Exponential decay per edge hop (0 < decay_rate <= 1).
            prune_thresh (float): Minimum flow threshold to retain paths.
            sim_should (float): Minimum cosine similarity to add edge.

        Raises:
            ValueError: On invalid parameter values.
            RuntimeError: If the embedding model fails to load.
        """
        if not (0 < decay_rate <= 1):
            raise ValueError("decay_rate must be in (0, 1]")
        if prune_thresh < 0:
            raise ValueError("prune_thresh must be non-negative")
        if not (0 <= sim_should <= 1):
            raise ValueError("sim_should must be in [0, 1]")

        try:
            self.embedding_model = embedding_model
        except Exception as e:
            logger.error("Failed to load embedding model '%s': %s", embedding_model, e)
            raise RuntimeError(f"Embedding model loading failed: {e}")

        self.conn = conn
        self.g = nx.DiGraph()
        self.decay_rate = decay_rate
        self.prune_thresh = prune_thresh
        self.sim_should = sim_should
        logger.info("Initialized PathRAG with decay=%.2f, threshold=%.3f, sim=%.2f",
                    decay_rate, prune_thresh, sim_should)

    def build_graph(self, max_workers: int = 8) -> None:
        self.g.clear()
        try:
            rows = pull_from_table(
                conn=self.conn,
                table_name="embed_vector",
                columns=["chunk", "embedding", "chunk_id"]
            )
        except Exception as e:
            logger.error("Failed to query embedding table: %s", e)
            raise RuntimeError(f"Database retrieval failed: {e}")

        vectors = []
        chunk_ids = []

        def process_row(row):
            try:
                chunk = row["chunk"]
                chunk_id = row["chunk_id"]
                emb_data = json.loads(row["embedding"])
                emb_array = np.array(emb_data, dtype=np.float32).flatten()
                return (chunk_id, chunk, emb_array)
            except Exception as e:
                logger.warning(f"Failed to process row {row.get('chunk_id')}: {e}")
                return None

        processed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row, row) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading nodes", ncols=80, colour="green"):
                result = future.result()
                if result:
                    processed.append(result)

        for chunk_id, chunk, emb_array in processed:
            self.g.add_node(chunk_id, text=chunk, emb=emb_array)
            chunk_ids.append(chunk_id)
            vectors.append(emb_array)

        # Edge creation with similarity threshold
        for i in tqdm(range(len(vectors)), desc="Building edges", ncols=80, colour="blue"):
            for j in range(i + 1, len(vectors)):
                try:
                    sim = cosine_similarity(
                        vectors[i].reshape(1, -1),
                        vectors[j].reshape(1, -1)
                    )[0, 0]
                    if sim > self.sim_should:
                        self.g.add_edge(chunk_ids[i], chunk_ids[j], weight=sim)
                        self.g.add_edge(chunk_ids[j], chunk_ids[i], weight=sim)
                except Exception as e:
                    logger.debug("Skipping edge between %s and %s: %s", chunk_ids[i], chunk_ids[j], e)

        logger.info("Graph built with %d nodes and %d edges",
                    self.g.number_of_nodes(), self.g.number_of_edges())

    def retrieve_nodes(self, query: str, top_k: int = 5) -> List[int]:
        """
        Retrieve top-k nodes in the graph most similar to a query.

        Args:
            query (str): Input user query.
            top_k (int): Number of most similar nodes to return.

        Returns:
            List[int]: List of node IDs sorted by similarity.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On embedding or similarity computation failure.
        """
        if not query:
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.g.number_of_nodes() == 0:
            raise RuntimeError("Graph is empty; run build_graph() first")

        try:
            # Get query embedding and ensure it's a numpy array
            q_emb = self.embedding_model.embed_texts(query, convert_to_numpy=True)
            
            # Handle case where embedding might be returned as list
            if isinstance(q_emb, list):
                q_emb = np.array(q_emb, dtype=np.float32)
            
            # Ensure proper shape (1D array)
            q_emb = q_emb.squeeze()  # This removes single-dimensional entries
            if q_emb.ndim != 1:
                raise ValueError(f"Expected 1D embedding, got shape {q_emb.shape}")
                
        except Exception as e:
            logger.error("Query embedding processing failed: %s", e)
            raise RuntimeError(f"Query embedding processing failed: {e}")

        sims = {}
        for nid, data in self.g.nodes(data=True):
            try:
                node_emb = data["emb"]
                # Ensure node embedding is numpy array
                if isinstance(node_emb, list):
                    node_emb = np.array(node_emb, dtype=np.float32)
                
                # Compute cosine similarity
                q_norm = q_emb / np.linalg.norm(q_emb)
                node_norm = node_emb / np.linalg.norm(node_emb)
                sim = np.dot(q_norm, node_norm)
                sims[nid] = float(sim)
                
            except Exception as e:
                logger.warning("Similarity computation failed for node %s: %s", nid, e)
                continue

        if not sims:
            raise RuntimeError("No valid similarities computed")

        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_ids = [nid for nid, _ in ranked]
        logger.info("Top %d nodes retrieved: %s", top_k, top_ids)
        return top_ids

    def prune_paths(self, nodes: List[int], max_hops: int = 4) -> List[List[int]]:
        if not nodes:
            raise ValueError("Node list is empty")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if any(n not in self.g for n in nodes):
            raise ValueError("Some nodes are missing in the graph")
        if self.g.number_of_edges() == 0:
            logger.warning("Graph has no edges - returning empty path list")
            return []

        paths = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                try:
                    # LIMIT paths for scalability
                    paths_gen = nx.all_simple_paths(self.g, u, v, cutoff=max_hops)
                    limited_paths = list(islice(paths_gen, 10))  # cap to 10 paths
                    for path in limited_paths:
                        flow = self._compute_flow(path)
                        if flow >= self.prune_thresh:
                            paths.append(path)
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning("Path finding failed for (%s, %s): %s", u, v, e)

        logger.info("Pruned %d paths from %d source nodes", len(paths), len(nodes))
        return paths

    def _compute_flow(self, path: List[int]) -> float:
        """
        Compute flow score for a path using edge weights and decay.

        Args:
            path (List[int]): Path of node IDs.

        Returns:
            float: Flow score, or 0 if computation fails.

        Raises:
            ValueError: If path is too short.
        """
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        try:
            weights = [
                self.g.edges[path[i], path[i + 1]]["weight"]
                for i in range(len(path) - 1)
            ]
            base_flow = np.prod(weights)
            return base_flow * (self.decay_rate ** (len(path) - 1))
        except Exception as e:
            logger.warning("Flow computation failed for path %s: %s", path, e)
            return 0.0

    def score_paths(self, paths: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Assign scores to paths using decay-aware flow scoring.

        Args:
            paths (List[List[int]]): List of node ID paths.

        Returns:
            List[Dict[str, Any]]: Sorted path dicts with 'path' and 'score'.

        Raises:
            ValueError: If path list is empty.
        """
        if not paths:
            raise ValueError("Empty paths list provided")

        scored = []
        for path in paths:
            try:
                score = self._compute_flow(path)
                scored.append({"path": path, "score": score})
            except ValueError as e:
                logger.warning("Skipping path %s: %s", path, e)

        scored.sort(key=lambda x: x["score"])
        if scored:
            logger.debug("Top path: %s with score %.3f", scored[-1]["path"], scored[-1]["score"])
        return scored

    def generate_prompt(self, query: str, scored_paths: List[Dict[str, Any]]) -> str:
        """
        Generate a RAG prompt from query and top scored paths.

        Args:
            query (str): User query string.
            scored_paths (List[Dict[str, Any]]): List of path-score dicts.

        Returns:
            str: Concatenated prompt string for language model.

        Raises:
            ValueError: On invalid input types or empty fields.
        """
        if not query:
            raise ValueError("Query must be non-empty")
        if not isinstance(scored_paths, list):
            raise ValueError("scored_paths must be a list")

        lines = ["QUERY: " + query, "RELATED EVIDENCE PATHS:"]
        for item in scored_paths:
            if not isinstance(item, dict) or "path" not in item:
                logger.warning("Skipping invalid scored path item: %s", item)
                continue
            try:
                texts = [self.g.nodes[nid]["text"] for nid in item["path"]]
                path_str = " → ".join(texts)
                lines.append(f"- [Score: {item['score']:.3f}] {path_str}")
            except Exception as e:
                logger.warning("Failed to process path %s: %s", item.get("path"), e)

        prompt = "\n".join(lines)
        logger.info("Generated prompt with %d paths (length: %d chars)",
                    len(scored_paths), len(prompt))
        return prompt

if __name__ == "__main__":
    from src.db import get_sqlite_engine
    from src.llms_providers import HuggingFaceModel

    # 1. Connect to SQLite and initialize embedding model
    conn = get_sqlite_engine()
    embedding_model = HuggingFaceModel(model_name="all-MiniLM-L6-v2")

    # 2. Initialize PathRAG
    rag = PathRAG(conn=conn, embedding_model=embedding_model)

    # 3. Build the semantic similarity graph from DB
    print("Building graph...")
    rag.build_graph()

    # 4. Define user query
    query = "What is machine learning?"

    # 5. Retrieve top relevant nodes
    top_nodes = rag.retrieve_nodes(query=query, top_k=5)
    print(f"Top retrieved node IDs: {top_nodes}")

    # 6. Attempt to prune and score relational paths
    paths = rag.prune_paths(top_nodes)
    print(f"Total paths found: {len(paths)}")

    if not paths:
        print("⚠️ No valid paths found between retrieved nodes.")
        logger.warning("Falling back to direct top-k node evidence.")

        # Fallback: Treat top-k nodes as individual evidence paths
        scored_paths = [{"path": [nid], "score": 1.0} for nid in top_nodes]

        print("\n⚠️ Using fallback direct node evidence:")
        for item in scored_paths:
            print(f"Node: {item['path'][0]} | Score: {item['score']:.3f}")
    else:
        # 7. Score and select top paths
        scored_paths = rag.score_paths(paths)
        top_k_paths = sorted(scored_paths, key=lambda x: x["score"], reverse=True)[:5]

        print("\nTop scored paths:")
        for path_obj in top_k_paths:
            print(f"Score: {path_obj['score']:.3f} | Path: {path_obj['path']}")

        scored_paths = top_k_paths

    # 8. Generate the final prompt
    prompt = rag.generate_prompt(query=query, scored_paths=scored_paths)

    # 9. Output prompt
    print("\n====== Generated Prompt ======\n")
    print(prompt)

    # Optional: Structured result
    result = {
        "top_k_paths": scored_paths,
        "prompt": prompt
    }
