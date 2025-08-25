"""
User File Upload API Route - FIXED VERSION
===========================================

This module provides a FastAPI route for handling user file uploads.  
It processes documents by saving, chunking, embedding, OCR (if PDF has images),
and building a PathRAG semantic graph. Additionally, file metadata is stored in MongoDB.

Key Fix: Extract text content from Document objects before embedding generation.

Features:
---------
- Validate file extension
- Save file to user-specific directory
- Extract images & text via OCR (if applicable)
- Chunk documents into smaller pieces
- Generate embeddings for chunks + OCR text
- Build and save PathRAG semantic graph
- Store metadata in MongoDB
"""

import os
import sys
import shutil
from pathlib import Path
import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pymongo import MongoClient

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src.controllers import (
    generate_unique_filename,
    AdvancedOCRProcessor,
    chunking_docs,
    ExtractionImagesFromPDF,
)
from src.utils import get_size
from src.rag import PathRAG
from src import get_path_rag, get_mongo_db, get_embedding_model
from src.llms_providers import HuggingFaceModel

# Initialize logger and settings
logger = setup_logging(name="USER-UPLOAD-FILE")
app_settings: Settings = get_settings()

user_file_route = APIRouter(
    prefix="/api/v1/user/file",
    tags=["File Uploads By user"],
    responses={404: {"description": "Not found"}},
)

UPLOAD_DIR = app_settings.DOC_LOCATION_STORE
ALLOWED_EXTENSIONS = app_settings.FILE_TYPES


def extract_text_from_chunks(chunks):
    """
    Extract text content from chunks, handling both Document objects and strings.
    
    Args:
        chunks: List of chunks (Document objects or strings)
    
    Returns:
        List of strings containing the text content
    """
    text_chunks = []
    
    for chunk in chunks:
        try:
            # If it's a Document object (from langchain), extract page_content
            if hasattr(chunk, 'page_content'):
                text_chunks.append(chunk.page_content)
            # If it's already a string, use it directly
            elif isinstance(chunk, str):
                text_chunks.append(chunk)
            # If it's a dict with 'content' or 'text' key
            elif isinstance(chunk, dict):
                if 'content' in chunk:
                    text_chunks.append(chunk['content'])
                elif 'text' in chunk:
                    text_chunks.append(chunk['text'])
                elif 'page_content' in chunk:
                    text_chunks.append(chunk['page_content'])
                else:
                    # Convert dict to string as fallback
                    text_chunks.append(str(chunk))
            else:
                # Convert other types to string
                text_chunks.append(str(chunk))
                
        except Exception as e:
            logger.warning(f"Failed to extract text from chunk: {e}")
            # Try to convert to string as fallback
            text_chunks.append(str(chunk))
    
    # Filter out empty strings
    text_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    
    return text_chunks


@user_file_route.post("", response_class=JSONResponse)
async def user_file(
    user_id: str,
    file: UploadFile = File(..., description="File to upload"),
    path_rag: PathRAG = Depends(get_path_rag),
    mongo_db: MongoClient = Depends(get_mongo_db),
    embedding_model: HuggingFaceModel = Depends(get_embedding_model),
):
    """
    Handle user file upload, OCR extraction, chunking, embedding, and PathRAG graph building.
    """
    try:
        logger.info(f"Received upload request: user={user_id}, file={file.filename}")

        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not allowed",
            )

        # Prepare user-specific save directory
        current_dir_name = user_id or Path(file.filename).stem
        sanitized_dir = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in current_dir_name
        )[:30]
        save_dir = os.path.join(UPLOAD_DIR, sanitized_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Save file with unique filename
        unique_filename = generate_unique_filename(file.filename)
        save_path = os.path.join(save_dir, unique_filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved at {save_path} ({get_size(save_path)} MB)")

        # OCR (if PDF with images)
        chunks_ocr = []
        if file_ext == ".pdf":
            try:
                images = ExtractionImagesFromPDF(pdf_path=save_path).extract_images()
                for img in images:
                    full_path = os.path.join(MAIN_DIR, img)
                    text = AdvancedOCRProcessor().extract_text(image=full_path)
                    if text:
                        chunks_ocr.extend(text if isinstance(text, list) else [text])
                logger.info(f"OCR extracted {len(chunks_ocr)} chunks from images")
            except Exception as ocr_err:
                logger.warning(f"OCR extraction failed: {ocr_err}")

        # Chunking
        chunks_meta = chunking_docs(file_path=save_path)
        chunks = chunks_meta.get("chunks", [])
        
        # FIXED: Extract text content from Document objects
        text_chunks = extract_text_from_chunks(chunks)
        
        if not text_chunks and not chunks_ocr:
            raise ValueError("No chunks produced from document or OCR")

        # Combine all text chunks (now all are strings)
        all_text_chunks = text_chunks + chunks_ocr
        
        logger.info(f"Total text chunks prepared for embedding: {len(all_text_chunks)}")

        # Embeddings - should work now with string inputs
        try:
            embeddings_vectors = embedding_model.embed_texts(texts=all_text_chunks)
            if not embeddings_vectors:
                raise ValueError("Failed to generate embeddings - no vectors returned")
            
            # Convert embeddings to numpy array if not already
            import numpy as np
            if not isinstance(embeddings_vectors, np.ndarray):
                # Handle different embedding formats
                if hasattr(embeddings_vectors, 'numpy'):
                    # PyTorch tensor
                    embeddings_vectors = embeddings_vectors.numpy()
                elif isinstance(embeddings_vectors, list):
                    # List of embeddings
                    embeddings_vectors = np.array(embeddings_vectors)
                else:
                    # Try direct conversion
                    embeddings_vectors = np.array(embeddings_vectors)
            
            logger.info(f"Generated embedding array with shape: {embeddings_vectors.shape}")
            
        except Exception as embed_err:
            logger.error(f"Embedding generation failed: {embed_err}")
            raise ValueError(f"Failed to generate embeddings: {embed_err}")

        # Build PathRAG graph
        graph_user_saved = os.path.join(MAIN_DIR, "storge_graph", f"{user_id}.pickle")
        os.makedirs(os.path.dirname(graph_user_saved), exist_ok=True)
        
        try:
            path_rag.build_graph(chunks=all_text_chunks, embeddings=embeddings_vectors)
            path_rag.save_graph(file_path=graph_user_saved)
            logger.info(f"PathRAG graph saved to {graph_user_saved}")
        except Exception as graph_err:
            logger.error(f"PathRAG graph building failed: {graph_err}")
            raise ValueError(f"Failed to build PathRAG graph: {graph_err}")

        # Store metadata in MongoDB
        try:
            db = mongo_db[app_settings.MONGO_DB_NAME]
            collection = db["user_files"]
            metadata = {
                "user_id": user_id,
                "filename": unique_filename,
                "file_path": save_path,
                "graph_path": graph_user_saved,
                "num_chunks": len(all_text_chunks),
                "num_ocr_chunks": len(chunks_ocr),
                "num_text_chunks": len(text_chunks),
            }
            collection.insert_one(metadata)
            logger.info("File metadata stored in MongoDB")
        except Exception as db_err:
            logger.warning(f"MongoDB insert failed: {db_err}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "File processed successfully",
                "file_name": unique_filename,
                "graph_path": graph_user_saved,
                "total_chunks": len(all_text_chunks),
                "text_chunks": len(text_chunks),
                "ocr_chunks": len(chunks_ocr),
            },
        )

    except Exception as e:
        logger.error(f"Failed to process file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )
