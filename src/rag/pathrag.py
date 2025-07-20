"""
PathRAG (Path-aware Retrieval-Augmented Generation)

Builds a semantic graph from text chunks, retrieves relevant subgraphs using query embeddings,
prunes paths using decay-flow scoring, and generates prompts with high-quality evidence paths.
"""

import heapq
import os
import sys
import json
import logging
from typing import List, Dict, Any
import numpy as np
import networkx as nx
from tqdm import tqdm
from sqlite3 import Connection
from sklearn.metrics.pairwise import cosine_similarity

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    print(f"Startup error: {e}")
    sys.exit(1)

from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.llms_providers import HuggingFaceModel

logger = setup_logging(name="PATH-RAG")
app_settings: Settings = get_settings()


class PathRAG:
    """
    Implements graph-based RAG reasoning using relational paths with decay-aware pruning.
    """

    def __init__(
        self,
        embedding_model: HuggingFaceModel,
        decay_rate: float = 0.8,
        prune_thresh: float = 0.01,
        sim_should: float = 0.1
    ) -> None:
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 and 1")
        if prune_thresh < 0:
            raise ValueError("prune_thresh must be non-negative")

        self.g = nx.DiGraph()
        self.decay_rate = decay_rate
        self.prune_thresh = prune_thresh
        self.sim_should = sim_should
        self.embedding_model = embedding_model

        logger.info("Initialized PathRAG (decay=%.2f, prune_thresh=%.3f, sim_thresh=%.3f)",
                    decay_rate, prune_thresh, sim_should)

    def build_graph(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Constructs a semantic graph using cosine similarity among chunk embeddings.
        """
        if not chunks or not isinstance(chunks, list):
            raise ValueError("Invalid or empty chunk list provided.")
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings size mismatch.")

        self.g.clear()
        logger.info("Building semantic graph for %d chunks...", len(chunks))

        try:
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self.g.add_node(idx, text=chunk, emb=emb)

            edge_count = 0
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
                        logger.warning("Similarity calc failed (%d, %d): %s", i, j, e)

            logger.info("Graph built: %d nodes, %d edges", self.g.number_of_nodes(), edge_count)
        except Exception as e:
            logger.error("Graph construction failed: %s", e)
            raise RuntimeError("Failed to build semantic graph.")

    def retrieve_nodes(self, query: str, top_k: int = 5) -> List[int]:
        """
        Retrieve top_k semantically similar nodes to the query.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive integer")
        if self.g.number_of_nodes() == 0:
            raise RuntimeError("Graph is empty - build_graph() first")

        try:
            q_emb = self.embedding_model.embed_texts(texts=query, convert_to_numpy=True)
            if isinstance(q_emb, list):
                q_emb = np.array(q_emb)
            if q_emb.ndim == 2 and q_emb.shape[0] == 1:
                q_emb = q_emb[0]

            if q_emb.ndim != 1:
                raise RuntimeError(f"Unexpected query embedding shape: {q_emb.shape}")

        except Exception as e:
            logger.error("Query embedding failed: %s", str(e))
            raise RuntimeError(f"Query embedding failed: {e}")

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
            raise RuntimeError("No valid node similarities computed")

        ranked = heapq.nlargest(top_k, sims.items(), key=lambda x: x[1])
        ranked = [nid for nid, _ in ranked]        
        logger.info("Top %d retrieved nodes: %s", top_k, ranked)
        return ranked

    def prune_paths(self, nodes: List[int], max_hops: int = 4) -> List[List[int]]:
        """
        Extracts paths between given nodes using decay-based scoring.
        """
        if not nodes or not isinstance(nodes, list):
            raise ValueError("Invalid node list.")
        if max_hops < 1:
            raise ValueError("max_hops must be ≥ 1")

        valid_paths = []
        logger.info("Starting path pruning over %d nodes (max_hops=%d)...", len(nodes), max_hops)

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
                        continue
                    except Exception as e:
                        logger.warning("Path search failed between %d and %d: %s", u, v, e)

        logger.info("Pruned to %d valid paths.", len(valid_paths))
        return valid_paths

    def _compute_flow(self, path: List[int]) -> float:
        """
        Computes decay-weighted flow score for a given path.
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
            if flow < self.prune_thresh:
                logger.debug("Pruned path %s with low flow %.6f", path, flow)
            return flow
        except Exception as e:
            logger.warning("Failed to compute flow for path %s: %s", path, e)
            return 0.0

    def score_paths(self, paths: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Scores paths and returns all sorted paths for prompt generation.
        """
        if not paths:
            raise ValueError("No paths to score.")

        results = []
        for p in paths:
            score = self._compute_flow(p)
            if score > 0:
                results.append({"path": p, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def generate_prompt(self, query: str, scored_paths: List[Dict[str, Any]]) -> str:
        """
        Converts top evidence paths into a prompt.
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
                logger.warning("Failed to format path %s: %s", item["path"], e)

        prompt = "\n".join(lines)
        logger.info("Prompt generated with %d paths (%d chars)", len(scored_paths), len(prompt))
        return prompt


def example_usage():
    from src.db import pull_from_table, get_sqlite_engine

    try:
        conn = get_sqlite_engine()
        embedding_model = HuggingFaceModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pathrag = PathRAG(
            embedding_model=embedding_model,
            decay_rate=0.85,
            prune_thresh=0.5,
            sim_should=0.5
        )
        rows = pull_from_table(conn=conn, table_name="embed_vector", columns=["chunk", "embedding"], limit=None)

        chunks, embeddings = [], []
        for row in tqdm(rows, desc="Parsing Chunks & Embeddings"):
            chunks.append(row["chunk"].strip().replace("\n", " "))
            emb_array = np.array(json.loads(row["embedding"]), dtype=np.float32)
            embeddings.append(emb_array)

        embeddings = np.vstack(embeddings)
        pathrag.build_graph(chunks, embeddings)

        while True:
            query = input("\nEnter your query (or type 'exit'): ").strip()
            if query.lower() == "exit":
                break

            nodes = pathrag.retrieve_nodes(query=query, top_k=5)
            paths = pathrag.prune_paths(nodes=nodes, max_hops=4)
            paths = pathrag.prune_paths(nodes=nodes, max_hops=4)
            if not paths:
                print("\n[!] No valid paths found. Try lowering prune_thresh or increasing max_hops.\n")
                continue

            scored_paths = pathrag.score_paths(paths)
            prompt = pathrag.generate_prompt(query, scored_paths)

            print("\n" + prompt + "\n")

    except Exception as e:
        logger.error("Error during example usage: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    example_usage()
