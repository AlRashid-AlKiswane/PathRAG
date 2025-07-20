"""
Utility module for interacting with SQLite tables, including robust logging and error handling.

Main Features:
- Dynamic data retrieval from SQLite tables.
- Structured logging of all DB operations.
- Simplified, clean API with minimal arguments.
"""

import logging
import os
import sys
import sqlite3
from typing import List, Optional, Dict, Any

# Setup logging and import custom log functions
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ModuleNotFoundError, ImportError) as e:
    logging.error("Failed to import logging utilities: %s", e, exc_info=True)
    raise
except Exception as e:
    logging.critical("Unexpected error during path setup: %s", e, exc_info=True)
    raise

from src.infra import setup_logging
logger = setup_logging()

def pull_from_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve rows from a specific SQLite table with an optional limit on the number of rows.

    Args:
        conn (sqlite3.Connection): Active SQLite database connection.
        table_name (str): Name of the table to query.
        columns (Optional[List[str]]): Columns to retrieve. Defaults to all columns if None.
        limit (Optional[int]): Number of rows to pull. If None, all rows will be retrieved.

    Returns:
        Optional[List[Dict[str, Any]]]: 
            A list of dictionaries representing each row, or None if an error occurs.

    Example:
        >>> pull_from_table(conn, "documents", ["id", "text"], limit=10)
        >>> [{'id': 1, 'text': 'example text'}, ...]
    """
    try:
        cursor = conn.cursor()

        # Use '*' if no columns are specified
        cols = ", ".join(columns) if columns else "*"
        
        # Create the base query
        query = f"SELECT {cols} FROM {table_name}"
        
        # Add limit to the query if specified
        if limit is not None:
            query += f" LIMIT {limit}"

        cursor.execute(query)

        # Get column names from cursor description
        col_names = [description[0] for description in cursor.description]
        
        # Fetch all the rows from the query result
        rows = cursor.fetchall()

        # Map column names to row values to create a list of dictionaries
        result = [dict(zip(col_names, row)) for row in rows]

        logger.info(f"Successfully pulled {len(result)} row(s) from '{table_name}'.")

        return result

    except sqlite3.OperationalError as e:
        logger.error(f"SQL error while querying '{table_name}': {e}")
    except sqlite3.DatabaseError as e:
        logger.error(f"Database error in '{table_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error pulling from '{table_name}': {e}")
    finally:
        logger.debug(f"Query execution completed for table '{table_name}'.")

    return None

if __name__ == "__main__":
    from src.db import get_sqlite_engine
    import pprint
    conn = get_sqlite_engine()

    data = pull_from_table(conn=conn,
                           table_name="chunks",
                           columns=["id", "chunk", "dataName"])
    
    print(data[:10])