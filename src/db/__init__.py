"""
Database Package Initialization

This package provides utilities for interacting with the SQLite database,
including engine creation, table initialization, data insertion, retrieval,
and clearing operations.

Modules included:
- db_enging: Contains the function to create and manage the SQLite engine.
- db_tables: Functions to initialize database tables (chunks, embed_vector, chatbot).
- db_insert: Functions for inserting records into the database tables.
- db_search: Functionality to retrieve data from tables.
- db_clear: Function to clear all records from specified tables.

Imports:
- get_sqlite_engine: Creates SQLite connection.
- init_chunks_table, init_embed_vector_table, init_chatbot_table: Initialize tables.
- insert_chunk, insert_embed_vector, insert_chatbot_entry: Insert data into tables.
- pull_from_table: Query data from tables.
- clear_table: Delete all data from a table.
"""

from .db_enging import get_sqlite_engine
from .db_search import pull_from_table
from .db_clear import clear_table
from .db_tables import (
    init_chunks_table,
    init_embed_vector_table,
    init_chatbot_table)
from .db_insert import (
    insert_chunk,
    insert_embed_vector,
    insert_chatbot_entry)

__all__ = [
    "get_sqlite_engine",
    "init_chunks_table",
    "init_embed_vector_table",
    "init_chatbot_table",
    "insert_chunk",
    "insert_embed_vector",
    "insert_chatbot_entry",
    "pull_from_table",
    "clear_table",
]


