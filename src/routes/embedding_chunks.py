"""
embedding_chunks_route.py

FastAPI route to convert textual document chunks into embeddings
and store them in MongoDB.

Route:
    POST /api/v1/chunks_to_embeddings/

Query Parameters:
    - fields (List[str]): Fields to retrieve from source collection. Default ["id", "chunk", "dataName"]
    - collection_name (str): Source collection to retrieve chunks from. Default "chunks"

Raises:
    - 404 if no chunks found.
    - 500 for DB or embedding errors.
    - 504 if embedding times out.

Author:
    ALRashid AlKiswane
"""

import asyncio
import os
import sys
import json
import uuid
from typing import List
from pymongo import MongoClient

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from tqdm import tqdm

# === Project Path Setup ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    import logging
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Project Imports ===
from src.llms_providers import HuggingFaceModel
from src.mongodb import pull_from_collection, insert_embed_vector_to_mongo
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_mongo_db, get_embedding_model

# === Logger & Settings ===
logger = setup_logging(name="EMBEDDING-CHUNKS")
app_settings: Settings = get_settings()

# === API Router ===
embedding_chunks_route = APIRouter(
    prefix="/api/v1/chunks_to_embeddings",
    tags=["Chunks → Embeddings"],
    responses={404: {"description": "Not found"}},
)

@embedding_chunks_route.post("", response_class=JSONResponse)
async def chunks_to_embeddings(
    fields: List[str] = ["id", "chunk", "dataName"],
    collection_name: str = "chunks",
    db: MongoClient = Depends(get_mongo_db),
    embedding_model: HuggingFaceModel = Depends(get_embedding_model),
) -> JSONResponse:
    """
    Retrieve text chunks from a MongoDB collection, generate embeddings,
    and store them in the 'embed_vector' collection.

    Args:
        fields (List[str]): Fields to retrieve from source collection.
        collection_name (str): Source MongoDB collection name.
        db (MongoClient): MongoDB database handle.
        embedding_model (HuggingFaceModel): Embedding model instance.

    Returns:
        JSONResponse: Number of chunks successfully embedded and stored.

    Raises:
        HTTPException: For missing chunks or internal errors.
    """
    try:
        logger.info("Retrieving chunks from collection '%s' with fields: %s", collection_name, fields)

        meta_chunks = pull_from_collection(
            db=db,
            collection_name=collection_name,
            fields=fields,
        )

        if not meta_chunks:
            logger.warning(f"No chunks found in collection '{collection_name}'.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found in collection '{collection_name}'."
            )

        valid_chunks = [row for row in meta_chunks if row.get("chunk")]
        if not valid_chunks:
            logger.warning("No usable chunk data found in retrieved documents.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No usable chunk data found."
            )

        logger.debug("%d valid chunks retrieved. Sample chunk start: %s", len(valid_chunks), valid_chunks[0]["chunk"][:100])

        processed_count = 0
        text_batch = []
        ids_batch = []

        # Process chunks one by one with tqdm progress bar
        for meta in tqdm(valid_chunks, desc="Embedding Chunks", unit="chunk"):
            chunk_text = meta["chunk"]
            chunk_id = str(meta.get("id") or uuid.uuid4())

            try:
                embedding_vector = embedding_model.embed_texts(
                    texts=chunk_text,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

                # Convert tensor or numpy array to list if needed
                if hasattr(embedding_vector, "tolist"):
                    embedding_vector = embedding_vector.tolist()

                serialized_embedding = json.dumps(embedding_vector)

                # Insert embedding into MongoDB
                success = insert_embed_vector_to_mongo(
                    db=db,
                    chunk=chunk_text,
                    embedding=serialized_embedding,
                    chunk_id=chunk_id,
                    doc_id=str(uuid.uuid4())
                )

                if success:
                    processed_count += 1

            except Exception as embed_err:
                tqdm.write(f"[ERROR] Embedding failed on chunk ID {chunk_id}: {embed_err}")
                logger.error(f"Embedding failed on chunk ID {chunk_id}: {embed_err}", exc_info=True)

        logger.info("Completed embedding: %d/%d chunks processed.", processed_count, len(valid_chunks))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Chunks successfully embedded and stored.",
                "chunks_processed": processed_count,
            }
        )

    except HTTPException as http_exc:
        logger.warning("HTTPException: %s", http_exc.detail)
        raise http_exc

    except asyncio.TimeoutError:
        logger.error("Embedding operation timed out.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding generation timed out."
        )

    except asyncio.CancelledError:
        logger.error("Embedding operation was cancelled.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding operation was cancelled."
        )

    except Exception as e:
        logger.exception("Unexpected error during chunk embedding.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during chunk embedding."
        ) from e
