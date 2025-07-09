"""
Chunking Route

This module defines a FastAPI route that accepts a file path or a directory name,
performs document chunking, and inserts the chunks into a SQLite database.

Features:
- Accepts an individual file path or a directory of documents
- Uses `chunking_docs` to split text into metadata chunks
- Inserts each chunk into the database via `insert_chunk`
- Handles file existence and I/O errors gracefully
"""

import os
import sys
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlite3 import Connection

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.db import insert_chunk, clear_table
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.controllers import chunking_docs
from src import get_db_conn

# Initialize logger and settings
logger = setup_logging()
app_settings: Settings = get_settings()

chunking_route = APIRouter(
    prefix="/api/v1/chunk",
    tags=["Chunking → Docs"],
    responses={404: {"description": "Not found"}}
)

@chunking_route.post("/", response_class=JSONResponse)
async def chunking(file_path: Optional[str] = None,
                   dir_file: Optional[str] = None,
                   reset_table: bool = False,
                   conn: Connection = Depends(get_db_conn)):
    """
    Process a single file or an entire directory for document chunking and store the chunks.

    Args:
        file_path (Optional[str]): Full path to a single document file to be chunked.
        dir_file (Optional[str]): Subdirectory inside 'assets/docs' to process all contained files.
        conn (Connection): Active SQLite connection injected by FastAPI.

    Returns:
        JSONResponse: Summary message indicating the success and total processed chunks.
    """
    total_chunks_inserted = 0

    if reset_table:
        clear_table(conn=conn,
                    table_name="chunks")
        logger.warning("Remove all Chunks in `chunks` table successfully.")

    # Single file processing
    if file_path:
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )
        logger.info("Chunking single file: %s", file_path)
        meta = chunking_docs(file_path=file_path)
        for chunk in meta["chunks"]:
            insert_chunk(conn=conn,
                         chunk=chunk.page_content,
                         file=file_path,
                         dataName=dir_file or "unknown")
            total_chunks_inserted += 1

    # Directory-based processing
    elif dir_file:
        dir_path = os.path.join(MAIN_DIR, "assets/docs", dir_file)
        if not os.path.exists(dir_path):
            logger.error("Directory not found: %s", dir_path)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {dir_path}"
            )
        logger.info("Processing directory: %s", dir_path)
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            if os.path.isfile(full_path):
                logger.debug("Chunking file in directory: %s", full_path)
                meta = chunking_docs(file_path=full_path)
                for chunk in meta["chunks"]:
                    insert_chunk(conn=conn,
                                 chunk=chunk.page_content,
                                 file=full_path,
                                 dataName=dir_file)
                    total_chunks_inserted += 1
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'dir_file' must be provided."
        )

    logger.info("Chunking completed. Total chunks inserted: %d", total_chunks_inserted)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Chunking successful", "total_chunks": total_chunks_inserted}
    )
