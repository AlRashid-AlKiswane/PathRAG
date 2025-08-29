"""
Markdown Chunking API Router Module

This module defines the FastAPI router for chunking markdown files and inserting
the resulting text chunks into a MongoDB collection. It enables efficient processing
of markdown content by splitting large files or directories of markdown files into
manageable chunks for downstream processing or retrieval.

Features:
- Supports chunking a single markdown file or recursively processing all markdown files
  in a given directory.
- Allows automatic clearing of the MongoDB `chunks` collection before insertion to
  avoid duplicate or stale data.
- Uses a custom MarkdownChunker class (from `src.controllers.md_files_chunking`) to
  perform chunking with configurable chunk size and overlap.
- Inserts chunks into MongoDB with progress tracked via `tqdm` progress bars.
- Includes robust error handling and logging at different levels to facilitate
  debugging and production monitoring.

Dependencies:
- `fastapi` for API routing and dependency injection.
- `pymongo` for MongoDB interaction.
- `tqdm` for progress bars.
- Internal project modules for chunking logic, database operations, settings, and schemas.

Usage:
- POST requests to `/api/v1/chunker-md` with JSON body specifying
  `input_path` (file or directory) and `recursive` (bool).
- Optional query parameter `do_reset` to clear chunks collection before processing.
- Returns detailed response with counts of total, inserted, and failed chunks.

Example Request Body:
{
  "input_path": "/path/to/markdown/files",
  "recursive": true
}

Example Response:
{
  "total_chunks": 120,
  "inserted_chunks": 120,
  "failed_chunks": 0,
  "message": "Chunking complete: 120 inserted, 0 failed."
}

Exceptions:
- Returns HTTP 404 if input path does not exist.
- Returns HTTP 400 if input path is invalid.
- Returns HTTP 500 on chunking or database errors.

Author: AlRashid
Date: 2025--08-11
"""


import os
import sys
from pathlib import Path

from pymongo import MongoClient
from fastapi import APIRouter, Depends, HTTPException, status
from tqdm import tqdm

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    print(f"Failed to configure project path: {e}")
    sys.exit(1)

# === Project Imports ===
from src.mongodb import insert_chunk_to_mongo, clear_collection
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.controllers import md_files_chunking
from src import get_mongo_db
from src.schemas import ChunkResponse, ChunkRequest

# === Logger and Settings ===
logger = setup_logging(name="CHUNKING-MD-FILES-ROUTE")
app_settings: Settings = get_settings()

# === API Router ===
md_chunker_routes = APIRouter(
    prefix="/api/v1/chunker-md",
    tags=["MD → CHUNK"],
    responses={404: {"description": "Not found"}}
)

@md_chunker_routes.post(
    "/",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk markdown files and insert chunks into MongoDB"
)
async def chunk_markdown_files(
    request: ChunkRequest,
    db: MongoClient = Depends(get_mongo_db),
    do_reset: bool = True
):
    """
    Endpoint to chunk markdown files from a specified path (file or directory),
    and insert resulting chunks into a MongoDB collection.

    This endpoint supports:
    - Processing a single markdown file or recursively processing all markdown files
      in a directory.
    - Optionally resetting (clearing) the MongoDB `chunks` collection before inserting
      new chunks.
    - Progress tracking for chunk insertion using tqdm.
    - Robust error handling with detailed logging.

    Args:
        request (ChunkRequest): Pydantic model containing:
            - input_path (str): Path to a markdown file or directory containing markdown files.
            - recursive (bool): Whether to recursively process markdown files in subdirectories.
        db (MongoClient, optional): MongoDB client dependency, injected by FastAPI.
        do_reset (bool, optional): Flag indicating whether to clear the `chunks` collection before inserting.
            Defaults to True.

    Returns:
        ChunkResponse: Contains total chunks found, how many were inserted successfully,
                       how many failed, and a status message.

    Raises:
        HTTPException 404: If the input path does not exist.
        HTTPException 400: If input path is neither a file nor a directory.
        HTTPException 500: For errors during chunking or database insertion.
    """
    logger.debug(f"Chunking request received: input_path={request.input_path}, recursive={request.recursive}, do_reset={do_reset}")

    if do_reset:
        try:
            logger.info("Resetting the 'chunks' collection before processing...")
            clear_collection(db=db, collection_name="chunks")
            logger.debug("'chunks' collection cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear 'chunks' collection: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reset chunks collection: {e}")

    path = Path(request.input_path)
    if not path.exists():
        logger.warning(f"Input path does not exist: {request.input_path}")
        raise HTTPException(status_code=404, detail=f"Input path does not exist: {request.input_path}")

    try:
        chunker = md_files_chunking.MarkdownChunker(chunk_size=500, chunk_overlap=50)
        if path.is_file():
            logger.info(f"Processing single markdown file: {request.input_path}")
            chunks = chunker.process_file(path)
        elif path.is_dir():
            logger.info(f"Processing markdown files in directory: {request.input_path}, recursive={request.recursive}")
            chunks = chunker.process_directory(path, recursive=request.recursive)
        else:
            logger.error(f"Invalid input path type (not file or directory): {request.input_path}")
            raise HTTPException(status_code=400, detail="Input path is neither a file nor a directory")
    except Exception as e:
        logger.error(f"Error during chunking markdown files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during chunking: {e}")

    if not chunks:
        logger.info("No chunks generated from the input path.")
        return ChunkResponse(
            total_chunks=0,
            inserted_chunks=0,
            failed_chunks=0,
            message="No chunks generated from the input path."
        )

    inserted = 0
    failed = 0
    try:
        logger.info(f"Inserting {len(chunks)} chunks into MongoDB...")
        for chunk_data in tqdm(chunks, desc="Inserting chunks", unit="chunk", leave=False):
            try:
                success = insert_chunk_to_mongo(
                    db=db,
                    chunk=chunk_data.get("chunk"),
                    file=chunk_data.get("file_path"),
                    data_name=chunk_data.get("title"),
                    size=chunk_data.get("size"),
                )
                if success:
                    inserted += 1
                else:
                    failed += 1
                    logger.debug(f"Chunk insertion failed for chunk with title: {chunk_data.get('title')}")
            except Exception as e:
                failed += 1
                logger.error(f"Exception during chunk insertion: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed during chunk insertion loop: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error inserting chunks into MongoDB: {e}")

    message = f"Chunking complete: {inserted} inserted, {failed} failed."
    logger.info(message)

    return ChunkResponse(
        total_chunks=len(chunks),
        inserted_chunks=inserted,
        failed_chunks=failed,
        message=message,
    )
