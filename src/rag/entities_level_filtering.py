"""
Module: entity_level_filtering.py
Description:
    Implements entity-level filtering to extract named entities from a query and retrieve
    associated chunk IDs from a SQLite database. Utilizes a provided NER model for entity extraction.
    Includes robust error handling and logging.
"""

import os
import sys
import logging
import sqlite3
from typing import List, Dict

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.db import pull_from_table
from src.llms_providers import NERModel

logger = setup_logging()


class EntityLevelFiltering:
    """
    Handles extraction of named entities from queries and retrieves associated chunk IDs
    from the database.

    Attributes:
        conn (sqlite3.Connection): SQLite database connection object.
        ner_model (NERModel): Named Entity Recognition model instance used for entity extraction.
    """

    def __init__(self,
                 conn: sqlite3.Connection,
                 ner_model: NERModel):
        """
        Initialize EntityLevelFiltering instance with database connection and NER model.

        Args:
            conn (sqlite3.Connection): SQLite database connection.
            ner_model (NERModel): NER model instance with a 'predict' method.
        """
        self.conn = conn
        self.ner_model = ner_model
        logger.info("EntityLevelFiltering initialized with database connection and NER model.")

    def _ner_query(self, query: str) -> List[Dict[str, str]]:
        """
        Extract named entities from the input query using the NER model.

        Args:
            query (str): Input query string.

        Returns:
            List[Dict[str, str]]: List of entities with 'text' and 'type' keys.

        Raises:
            RuntimeError: If the NER model prediction fails.
        """
        logger.debug(f"Starting NER query extraction for input: {query}")
        try:
            result = self.ner_model.predict(text=query)
            if not isinstance(result, dict):
                logger.warning(f"Unexpected NER model output type: {type(result)}. Expected list.")
                return []
            logger.info(f"Extracted {len(result)} entities from query.")
            logger.debug(f"Entities extracted: {result}")
            return result
        except Exception as e:
            logger.error(f"NER model prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"NER prediction failed: {e}")

    def _fetch_chunks_from_db(self, text: str, entity_type: str, top_k: int) -> List[Dict[str, str]]:
        """
        Retrieve top_k chunk texts from the database associated with the entity text or type.

        Args:
            text (str): Named entity text to match (partial match).
            entity_type (str): Entity type to match exactly.
            top_k (int): Maximum number of results to retrieve.

        Returns:
            List[Dict[str, str]]: List of dicts each containing a 'chunk' string.

        Raises:
            sqlite3.DatabaseError: If database query fails.
        """
        logger.debug(f"Fetching top_k={top_k} for text='{text}' OR type='{entity_type}'")
        result = []
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT source_text
                FROM (
                    SELECT source_text
                    FROM ner_entities
                    WHERE text LIKE ? OR type = ?
                    GROUP BY source_text
                )
                LIMIT ?
            """, (f"%{text}%", entity_type, top_k))
            rows = cursor.fetchall()
            logger.debug(f"Returned {len(rows)} rows: {[row[0] for row in rows]}")
            for row in rows:
                result.append({"chunk": row[0]})
            return result
        except sqlite3.DatabaseError as e:
            logger.error("Failed to fetch from DB: %s", e, exc_info=True)
            raise

    def entities_retrieval(self, query: str, top_k: int) -> List[Dict[str, int]]:
        """
        Extract entities from the query and retrieve unique associated chunk IDs from the database.

        Args:
            query (str): Query string to process.

        Returns:
            List[Dict[str, int]]: List of unique chunk_id dicts corresponding to entities.

        Raises:
            RuntimeError: If entity extraction or database queries fail.
        """
        logger.info(f"Starting entity retrieval for query: '{query}'")
        try:
            extracted_entities = self._ner_query(query=query)
            all_chunks = []
            for ent in extracted_entities["entities"]:
                logger.debug(f"Processing entity: {ent}")
                chunks = self._fetch_chunks_from_db(text=ent.get("text", ""), 
                                                    entity_type=ent.get("type", ""),
                                                    top_k=top_k)
                logger.debug(f"Chunks found for entity '{ent.get('text', '')}': {chunks}")
                all_chunks.extend(chunks)
            # Remove duplicates
            unique_chunks = {chunk["chunk"]: chunk for chunk in all_chunks}.values()
            unique_chunks_list = list(unique_chunks)
            logger.info(f"Total unique chunks retrieved: {len(unique_chunks_list)}")
            return unique_chunks_list
        except Exception as e:
            logger.error(f"Entity retrieval process failed: {e}", exc_info=True)
            raise RuntimeError(f"Entity retrieval failed: {e}")
