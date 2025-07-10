"""
faiss_rag.py

This module implements a lightweight FAISS-based retrieval system to perform
semantic search over vector embeddings stored in a SQLite database.
It includes full logging support with all severity levels:
DEBUG, INFO, WARNING, ERROR.
"""

import os
import sys
import logging
import sqlite3
import json
import numpy as np
import faiss

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.db import pull_from_table

logger = setup_logging()


class FaissRAG:
    """
    Lightweight FAISS-based semantic retriever using SQLite-stored embedding vectors.

    Attributes:
        conn (sqlite3.Connection): SQLite connection.
        vectors_embedding (np.ndarray): Matrix of embedding vectors.
        chunk_ids (list[int]): Corresponding chunk IDs.
        index (faiss.IndexFlatL2): FAISS index built from embeddings.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Initialize FaissRAG with a database connection.

        Args:
            conn (sqlite3.Connection): SQLite database connection.
        """
        self.conn = conn
        logger.info("FaissRAG initialized with active database connection.")

        self.vectors_embedding, self.chunk_ids = self._fetch_embedding_vectors()
        self.index = self._build_faiss_index()

    def _fetch_embedding_vectors(self) -> tuple[np.ndarray, list[int]]:
        """
        Fetch and decode embedding vectors from the database.

        Returns:
            tuple: (vectors_embedding: np.ndarray, chunk_ids: list[int])

        Raises:
            ValueError: If no embeddings are found or malformed.
        """
        try:
            logger.debug("Fetching embedding vectors from table 'embed_vector'.")

            records = pull_from_table(
                conn=self.conn,
                table_name="embed_vector",
                columns=["chunk_id", "embedding"]
            )

            chunk_ids = []
            vectors = []

            for row in records:
                chunk_id = row["chunk_id"]
                try:
                    emb_list = json.loads(row["embedding"])
                    emb_array = np.array(emb_list, dtype=np.float32)
                    vectors.append(emb_array)
                    chunk_ids.append(chunk_id)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed embedding for chunk_id={chunk_id}")

            if not vectors:
                raise ValueError("No valid embedding vectors found in the database.")

            vectors_embedding = np.vstack(vectors)
            logger.info("Successfully loaded %d embedding vectors.", len(vectors))

            return vectors_embedding, chunk_ids

        except Exception as e:
            logger.exception("Failed to load embeddings: %s", e)
            raise

    def _build_faiss_index(self) -> faiss.IndexFlatL2:
        """
        Build a FAISS index using L2 distance metric.

        Returns:
            faiss.IndexFlatL2: FAISS index instance.

        Raises:
            RuntimeError: If building index fails.
        """
        try:
            logger.debug("Building FAISS index using L2 distance.")

            _, dim = self.vectors_embedding.shape
            index = faiss.IndexFlatL2(dim)
            index.add(self.vectors_embedding)

            logger.info("FAISS index successfully built and populated.")
            return index

        except Exception as e:
            logger.exception("Error building FAISS index: %s", e)
            raise RuntimeError("Failed to build FAISS index.")

    def semantic_retrieval(self, embed_query: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Perform semantic retrieval using the FAISS index.

        Args:
            embed_query (np.ndarray): 1D query embedding vector.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of retrieved chunks with `id` and `chunk`.

        Raises:
            ValueError: If input dimensions are invalid.
        """
        logger.info("Performing semantic retrieval.")

        try:
            if embed_query.ndim != 1:
                raise ValueError("Query embedding must be 1D.")

            if embed_query.shape[0] != self.vectors_embedding.shape[1]:
                raise ValueError("Embedding dimension mismatch.")

            distances, indices = self.index.search(np.expand_dims(embed_query, axis=0), top_k)
            logger.debug("FAISS search complete. Retrieved indices: %s", indices[0])

            return self._fetch_chunks(indices[0])

        except Exception as e:
            logger.exception("Semantic retrieval failed: %s", e)
            raise

    def _fetch_chunks(self, indices: list[int]) -> list[dict]:
        """
        Retrieve chunk texts from the `chunks` table based on FAISS indices.

        Args:
            indices (list[int]): FAISS index results.

        Returns:
            list[dict]: List of chunks with `id` and `chunk`.

        Raises:
            sqlite3.Error: On database errors.
        """
        try:
            logger.debug("Fetching chunk texts based on retrieved indices.")
            cursor = self.conn.cursor()
            results = []

            for idx in indices:
                if 0 <= idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    cursor.execute("SELECT id, chunk FROM chunks WHERE id = ?", (chunk_id,))
                    row = cursor.fetchone()
                    if row:
                        results.append({"id": row[0], "chunk": row[1]})
                        logger.debug("Retrieved chunk id=%s", row[0])
                    else:
                        logger.warning("Chunk not found for id=%d", chunk_id)
                else:
                    logger.warning("Invalid FAISS index: %d", idx)

            logger.info("Retrieved %d chunks from the database.", len(results))
            return results

        except sqlite3.Error as e:
            logger.exception("Database error during chunk fetch: %s", e)
            raise
