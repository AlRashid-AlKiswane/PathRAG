"""
Chunk Insertion Module for RAG Database

This module provides functionality to insert individual document chunks into the
`chunks` table of an SQLite database. It is part of the document preprocessing
pipeline for a Retrieval-Augmented Generation (RAG) application.

Key Features:
- Inserts a document chunk along with its source file and data group name
- Handles database commit and rollback
- Uses lazy logging for efficient and clear runtime feedback
- Robust error handling to prevent data corruption

Functions:
- insert_chunk(): Safely inserts a single chunk into the database

Intended Usage:
- Used after document parsing and chunking to persist data
- Can be integrated with pipelines or called standalone for testing

Dependencies:
- SQLite3
- Custom settings and logging utilities from `src.helpers` and `src.infra`

Example:
    from src.db import insert_chunk
    conn = get_sqlite_engine()
    insert_chunk(conn, "chunk text", "file.pdf", "DocumentGroup")
"""


import logging
import os
import sys
import sqlite3
from typing import Optional

# Set up the main directory for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.getLogger(__name__).error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging

logger = setup_logging()

def insert_chunk(conn: sqlite3.Connection,
                 chunk: str = None,
                 file: str = None,
                 dataName: str = None) -> bool:
    """
    Inserts a document chunk into the 'chunks' table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        chunk (str): The text content of the chunk.
        file (str): The file path or name from which the chunk was extracted.
        dataName (str): The source or logical group the chunk belongs to.

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (chunk, file, dataName)
            VALUES (:chunk, :file, :dataName)
            """,
            {"chunk": chunk, "file": file, "dataName": dataName}
        )
        conn.commit()
        logger.info("Inserted chunk into 'chunks' table from file: %s", file)
        return True

    except sqlite3.Error as e:
        logger.error("Failed to insert chunk: %s", e)
        conn.rollback()
        return False

def insert_embed_vector(conn: sqlite3.Connection,
                        chunk: str,
                        embedding: str,
                        chunk_id: str) -> bool:
    """
    Inserts a document chunk and its embedding into the 'embed_vector' table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        chunk (str): The text content of the document chunk.
        embedding (str): Serialized embedding vector (e.g., JSON string).
        chunk_id (str): Unique identifier for the chunk (e.g., UUID or hash).

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO embed_vector (chunk, embedding, chunk_id)
            VALUES (:chunk, :embedding, :chunk_id)
            """,
            {"chunk": chunk, "embedding": embedding, "chunk_id": chunk_id}
        )
        conn.commit()
        logger.info("Inserted chunk with chunk_id: %s into 'embed_vector' table.", chunk_id)
        return True

    except sqlite3.Error as e:
        logger.error("Failed to insert into 'embed_vector' table: %s", e)
        conn.rollback()
        return False

def insert_ner_entity(
    conn: sqlite3.Connection,
    chunk_id: Optional[str],
    text: str,
    entity_type: str,
    start: int,
    end: int,
    score: float,
    source_text: str
) -> bool:
    """
    Insert a single named entity record into the 'ner_entities' table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        chunk_id (Optional[str]): Reference to the source chunk (can be None).
        text (str): Extracted entity text.
        entity_type (str): Type/category of the entity (e.g., 'PER', 'ORG').
        start (int): Character start index in source text.
        end (int): Character end index in source text.
        score (float): Confidence score of the prediction.
        source_text (str): Full text that contained the entity.

    Returns:
        bool: True if insertion is successful, False otherwise.

    Raises:
        sqlite3.Error: If insertion fails due to database error.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ner_entities (
                chunk_id, text, type, start, end, score, source_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, text, entity_type, start, end, score, source_text)
        )
        conn.commit()
        logger.debug("✅ Inserted NER entity: %s (%s)", text, entity_type)
        return True

    except sqlite3.Error as e:
        logger.exception("❌ Failed to insert NER entity: %s", e)
        return False

def insert_chatbot_entry(conn: sqlite3.Connection,
                         user_id: str = None,
                         query: str = None,
                         llm_response: str = None,
                         retrieval_context: str = None,
                         retrieval_rank: int = None) -> bool:
    """
    Inserts a new entry into the 'chatbot' table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        user_id (str): Identifier for the user who made the query (optional).
        query (str): The user's query. Required.
        llm_response (str): The response generated by the LLM. Required.
        retrieval_context (str): The text content used during retrieval. Required.
        retrieval_rank (int): The numerical rank or order of the retrieved context (optional).

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    try:
        if not query or not llm_response or not retrieval_context:
            logger.warning("Missing required fields: query, llm_response, or retrieval_context.")
            return False

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chatbot (user_id, query, llm_response, retrieval_context, retrieval_rank)
            VALUES (:user_id, :query, :llm_response, :retrieval_context, :retrieval_rank)
            """,
            {
                "user_id": user_id,
                "query": query,
                "llm_response": llm_response,
                "retrieval_context": retrieval_context,
                "retrieval_rank": retrieval_rank
            }
        )
        conn.commit()
        logger.info("✅ Inserted chatbot entry for user_id: %s", user_id)
        return True

    except sqlite3.Error as e:
        logger.error("❌ Failed to insert chatbot entry: %s", e)
        conn.rollback()
        return False


if __name__ == "__main__":
    from src.db import get_sqlite_engine

    conn = get_sqlite_engine()
    if conn:
        success = insert_chunk(
            conn=conn,
            chunk="Sample chunk text",
            file="example.pdf",
            dataName="ImmigrationGuide2025"
        )
        logger.info("Insertion success: %s", success)
        conn.close()

