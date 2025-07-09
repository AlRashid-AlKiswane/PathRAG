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
    A simple FAISS-based semantic retriever using embedding vectors stored in a database.

    Attributes:
        conn (sqlite3.Connection): SQLite database connection.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Initialize the FaissRAG object.

        Args:
            conn (sqlite3.Connection): SQLite database connection.
        """
        self.conn = conn
        logger.info("FaissRAG initialized with active database connection.")

    def _fetch_embedding_vector(self):
        """
        Fetches embedding vectors and corresponding chunk IDs from the database.

        Returns:
            tuple: (vectors_embed: np.ndarray, chunk_ids: list of int)

        Raises:
            ValueError: If no embeddings are found or conversion fails.
        """
        try:
            logger.debug("Attempting to pull embedding vectors from the database.")

            embedd_meta = pull_from_table(
                conn=self.conn,
                table_name="embed_vector",
                columns=["chunk_id", "embedding"]
            )

            chunk_ids = []
            embedding_vectors = []

            for i, row in enumerate(embedd_meta):
                logger.debug(f"Parsing embedding for chunk_id={row['chunk_id']}")
                chunk_ids.append(row["chunk_id"])
                embedding_blob = row["embedding"]

                try:
                    embedding_list = json.loads(embedding_blob)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON for chunk_id={row['chunk_id']}")
                    continue

                embedding_array = np.array(embedding_list, dtype=np.float32)
                embedding_vectors.append(embedding_array)

            if not embedding_vectors:
                logger.error("No valid embedding vectors found.")
                raise ValueError("No embedding vectors found in the database.")

            vectors_embed = np.vstack(embedding_vectors)
            logger.info(f"Successfully loaded {len(embedding_vectors)} embedding vectors.")

            return vectors_embed, chunk_ids

        except Exception as e:
            logger.exception("Failed to fetch or parse embeddings: %s", e)
            raise

    def _build_faiss_index(self, vectors_embedding: np.ndarray) -> faiss.IndexFlatL2:
        """
        Builds a FAISS index using L2 distance.

        Args:
            vectors_embedding (np.ndarray): Array of embedding vectors.

        Returns:
            faiss.IndexFlatL2: FAISS index built on embeddings.
        """
        try:
            logger.debug("Building FAISS index.")
            _, dim = vectors_embedding.shape
            index = faiss.IndexFlatL2(dim)
            index.add(vectors_embedding)
            logger.info("FAISS index built and populated.")
            return index
        except Exception as e:
            logger.exception("Failed to build FAISS index: %s", e)
            raise

    def semantic_retrieval(self, embed_query: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search to retrieve the most relevant chunks.

        Args:
            embed_query (np.ndarray): Embedding vector of the query.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of retrieved chunks with IDs and content.
        """
        logger.info("Starting semantic retrieval.")
        try:
            if embed_query.ndim != 1:
                logger.error("Query embedding must be a 1D array.")
                raise ValueError("Query embedding must be 1D.")

            vectors_embed, chunk_ids = self._fetch_embedding_vector()

            if embed_query.shape[0] != vectors_embed.shape[1]:
                logger.error("Embedding dimension mismatch.")
                raise ValueError("Query embedding dimension mismatch.")

            index = self._build_faiss_index(vectors_embed)
            logger.debug("Searching FAISS index.")

            distances, indices = index.search(np.expand_dims(embed_query, axis=0), top_k)

            logger.info(f"Top-{top_k} retrieval completed.")
            return self._fetch_chunk_retrieval(indices[0], chunk_ids)

        except Exception as e:
            logger.exception("Semantic retrieval failed: %s", e)
            raise

    def _fetch_chunk_retrieval(self, indices, chunk_ids):
        """
        Fetch chunk texts from the database using retrieved indices.

        Args:
            indices (list[int]): Indices returned by FAISS search.
            chunk_ids (list[int]): Corresponding chunk IDs.

        Returns:
            list[dict]: Retrieved chunks as dictionaries with `id` and `chunk`.
        """
        try:
            logger.debug("Fetching chunk texts for retrieved indices.")
            cursor = self.conn.cursor()
            results = []

            for idx in indices:
                if idx < 0 or idx >= len(chunk_ids):
                    logger.warning(f"Invalid index encountered during retrieval: {idx}")
                    continue

                chunk_id = chunk_ids[idx]
                cursor.execute("SELECT id, chunk FROM chunks WHERE id = ?", (chunk_id,))
                row = cursor.fetchone()

                if row:
                    results.append({"id": row[0], "chunk": row[1]})
                    logger.debug(f"Retrieved chunk: id={row[0]}")
                else:
                    logger.warning(f"No chunk found with id={chunk_id}")

            logger.info(f"Retrieved {len(results)} valid chunks.")
            return results

        except sqlite3.Error as e:
            logger.exception("Database error while fetching chunks: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error during chunk retrieval: %s", e)
            raise
