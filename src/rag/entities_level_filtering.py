"""
Module: entity_level_filtering.py

Description:
    Implements entity-level filtering by extracting named entities from a natural language
    query using an NER model and retrieving associated chunk IDs from a SQLite database.
    It performs precise filtering using Python-side JSON parsing due to SQLite limitations.

Author: AlRashid AlKiswane
Date: 2025-07-15
"""

import json
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
    logging.critical("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

from src.infra import setup_logging
from src.db import pull_from_table
from src.llms_providers import NERModel

logger = setup_logging()


class EntityLevelFiltering:
    """
    Handles extraction of named entities from queries and retrieves associated chunk IDs
    from a SQLite database.

    Attributes:
        conn (sqlite3.Connection): SQLite database connection object.
        ner_model (NERModel): Named Entity Recognition model instance.
    """

    def __init__(self, conn: sqlite3.Connection, ner_model: NERModel):
        """
        Initialize the EntityLevelFiltering instance.

        Args:
            conn (sqlite3.Connection): SQLite connection object.
            ner_model (NERModel): Named Entity Recognition model with a `predict` method.
        """
        self.conn = conn
        self.ner_model = ner_model
        logger.info("✅ EntityLevelFiltering initialized.")

    def _ner_query(self, query: str) -> List[str]:
        """
        Extract named entities from a given query using the NER model.

        Args:
            query (str): Natural language input query.

        Returns:
            List[str]: List of unique entity strings extracted from the query.

        Raises:
            RuntimeError: If the NER model fails or returns unexpected output.
        """
        logger.debug(f"🔍 Extracting NER entities from query: {query}")
        try:
            extracted = self.ner_model.predict(text=query)
            if not isinstance(extracted, list):
                logger.warning("⚠️ NER model output is not a list. Returning empty entity list.")
                return []
            logger.info(f"✅ Extracted {len(extracted)} entities: {extracted}")
            return extracted
        except Exception as e:
            logger.error(f"❌ NER model prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"NER prediction failed: {e}")

    def entities_retrieval(self, query: str, top_k: int) -> List[int]:
        """
        Retrieve chunk IDs from the database based on entities extracted from the query.

        Args:
            query (str): The input query string.
            top_k (int): Maximum number of chunks to retrieve per entity.

        Returns:
            List[int]: List of unique chunk IDs associated with the extracted entities.

        Raises:
            RuntimeError: If database access or entity extraction fails.
        """
        logger.info(f"🚀 Starting entity-level retrieval for query: '{query}'")
        try:
            extracted_entities = self._ner_query(query=query)
            if not extracted_entities:
                logger.warning("⚠️ No entities extracted. Returning empty result.")
                return []

            # Pull all rows from ner_entities (chunk_id, entities JSON)
            try:
                rows = pull_from_table(
                    conn=self.conn,
                    columns=["chunk_id", "entities"],
                    table_name="ner_entities"
                )
                logger.debug(f"📦 Retrieved {len(rows)} rows from 'ner_entities' table.")
            except Exception as db_error:
                logger.error(f"❌ Failed to fetch data from database: {db_error}", exc_info=True)
                raise RuntimeError(f"Database query failed: {db_error}")

            # Filter chunks by exact entity match
            all_chunk_ids = set()

            for entity in extracted_entities:
                logger.debug(f"🔎 Searching for chunks containing entity: '{entity}'")
                matched_count = 0

                for chunk_id, entity_json in rows:
                    try:
                        entity_list = json.loads(entity_json)
                        if not isinstance(entity_list, list):
                            logger.warning(f"Skipping malformed entity list for chunk {chunk_id}")
                            continue
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping chunk {chunk_id} due to JSON decode error.")
                        continue

                    if entity in entity_list:
                        all_chunk_ids.add(chunk_id)
                        matched_count += 1
                        logger.debug(f"✅ Match found in chunk {chunk_id} for entity '{entity}'")
                        if matched_count >= top_k:
                            break  # Stop if top_k matches for this entity

            logger.info(f"✅ Total unique chunks retrieved: {len(all_chunk_ids)}")
            return list(all_chunk_ids)

        except Exception as e:
            logger.error(f"❌ Entity retrieval process failed: {e}", exc_info=True)
            raise RuntimeError(f"Entity retrieval failed: {e}")
