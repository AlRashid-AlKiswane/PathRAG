"""
chunking_route.py

API route for chunking documents from single files or directories,
inserting chunks into MongoDB, with optional table reset.

Route:
    POST /api/v1/chunk/

Query Params:
    - file_path (str, optional): Full path to single document file.
    - dir_file (str, optional): Subdirectory inside 'assets/docs' to process all files.
    - reset_table (bool, default=False): Clear 'chunks' collection before inserting.

Raises:
    - 404 if file or directory not found.
    - 400 if neither file_path nor dir_file provided.

Author:
    ALRashid AlKiswane
"""

import os
import sys
import uuid
from typing import Optional
from pymongo import MongoClient

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
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
from src.graph_db import insert_chunk_to_mongo, clear_collection
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.controllers import chunking_docs, TextCleaner
from src import get_mongo_db

# === Logger and Settings ===
logger = setup_logging(name="CHUNKING-DOCS")
app_settings: Settings = get_settings()

# === API Router ===
chunking_route = APIRouter(
    prefix="/api/v1/chunk",
    tags=["Chunking → Docs"],
    responses={404: {"description": "Not found"}}
)

@chunking_route.post("/", response_class=JSONResponse)
async def chunking(
    file_path: Optional[str] = None,
    dir_file: Optional[str] = None,
    reset_table: bool = False,
    db: MongoClient = Depends(get_mongo_db)
) -> JSONResponse:
    """
    Chunk a single file or all files in a directory and store chunks in MongoDB.

    Args:
        file_path (Optional[str]): Path to single document file to chunk.
        dir_file (Optional[str]): Subdirectory inside 'assets/docs' for batch chunking.
        reset_table (bool): If True, clears 'chunks' collection before inserting.
        db (MongoClient): MongoDB database handle injected by FastAPI dependency.

    Returns:
        JSONResponse: Message with success status and total chunks inserted.

    Raises:
        HTTPException: If input parameters are invalid or files/directories not found.
    """
    total_chunks_inserted = 0

    if reset_table:
        try:
            clear_collection(db=db, collection_name="chunks")
            logger.info("Cleared 'chunks' collection before inserting new chunks.")
        except Exception as e:
            logger.error(f"Failed to clear 'chunks' collection: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to reset chunks collection.")

    text_cleaner = TextCleaner(lowercase=True)

    # Process a single file
    if file_path:
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )

        logger.info(f"Starting chunking for single file: {file_path}")
        try:
            meta = chunking_docs(file_path=file_path)
            for chunk in tqdm(meta["chunks"], desc="Chunking Single File", unit="chunk"):
                cleaned_text = text_cleaner.clean(chunk.page_content)
                doc_id = str(uuid.uuid4())
                insert_chunk_to_mongo(
                    db=db,
                    chunk=cleaned_text,
                    file=file_path,
                    data_name="single_file",
                    doc_id=doc_id
                )
                total_chunks_inserted += 1
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Chunking failed for file {file_path}")

    # Process all files in directory
    elif dir_file:
        dir_path = os.path.join(MAIN_DIR, "assets/docs", dir_file)
        if not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {dir_path}"
            )

        logger.info(f"Starting chunking for directory: {dir_path}")
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for filename in tqdm(files, desc="Chunking Files in Directory", unit="file"):
            full_path = os.path.join(dir_path, filename)
            try:
                meta = chunking_docs(file_path=full_path)
                for chunk in meta["chunks"]:
                    cleaned_text = text_cleaner.clean(chunk.page_content)
                    doc_id = str(uuid.uuid4())
                    insert_chunk_to_mongo(
                        db=db,
                        chunk=cleaned_text,
                        file=full_path,
                        data_name=dir_file,
                        doc_id=doc_id
                    )
                    total_chunks_inserted += 1
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to process file {filename}: {e}")
                logger.error(f"Failed to chunk file {filename} in directory {dir_path}: {e}", exc_info=True)

    else:
        logger.error("Neither 'file_path' nor 'dir_file' was provided.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'dir_file' must be provided."
        )

    logger.info(f"Chunking completed. Total chunks inserted: {total_chunks_inserted}")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Chunking successful",
            "total_chunks": total_chunks_inserted
        }
    )
