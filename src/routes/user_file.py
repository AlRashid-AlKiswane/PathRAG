"""
User File Upload API Route - COMPLETE FIXED VERSION
===================================================

This module provides a FastAPI route for handling user file uploads.  
It processes documents by saving, chunking, embedding, OCR (if PDF has images),
and building a PathRAG semantic graph. Additionally, file metadata is stored in MongoDB.

Key Fixes:
- Extract text content from Document objects before embedding generation
- Fixed OCR result handling (extract .text attribute from OCRResult objects)
- Proper error handling and validation
- Consistent text chunk processing
- Better logging and debugging information

Features:
---------
- Validate file extension and file size
- Save file to user-specific directory  
- Extract images & text via OCR (if applicable)
- Chunk documents into smaller pieces
- Generate embeddings for chunks + OCR text
- Build and save PathRAG semantic graph
- Store metadata in MongoDB with proper error handling
"""

import os
import sys
import shutil
from pathlib import Path
import logging
from typing import List, Union, Any
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
from src.utils import get_size, AutoSave
from src.rag import PathRAG
from src import get_path_rag, get_mongo_db, get_embedding_model
from src.llms_providers import HuggingFaceModel
from src.schemas import OCREngine

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
MAX_FILE_SIZE_MB = getattr(app_settings, 'MAX_FILE_SIZE_MB', 50)  # 50MB default


def extract_text_from_chunks(chunks: List[Any]) -> List[str]:
    """
    Extract text content from chunks, handling both Document objects and strings.
    
    Args:
        chunks: List of chunks (Document objects, strings, or dicts)
    
    Returns:
        List of strings containing the text content
    """
    text_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            # If it's a Document object (from langchain), extract page_content
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content.strip()
                if content:
                    text_chunks.append(content)
                    
            # If it's already a string, use it directly
            elif isinstance(chunk, str):
                content = chunk.strip()
                if content:
                    text_chunks.append(content)
                    
            # If it's a dict with content keys
            elif isinstance(chunk, dict):
                content = None
                for key in ['content', 'text', 'page_content']:
                    if key in chunk and chunk[key]:
                        content = str(chunk[key]).strip()
                        break
                
                if content:
                    text_chunks.append(content)
                else:
                    # Convert dict to string as fallback
                    dict_str = str(chunk).strip()
                    if dict_str and dict_str not in ['{}', 'None']:
                        text_chunks.append(dict_str)
                        
            else:
                # Convert other types to string
                content = str(chunk).strip()
                if content and content not in ['None', 'null', '']:
                    text_chunks.append(content)
                    
        except Exception as e:
            logger.warning(f"Failed to extract text from chunk {i}: {e}")
            # Try to convert to string as fallback
            try:
                fallback_content = str(chunk).strip()
                if fallback_content and fallback_content not in ['None', 'null', '']:
                    text_chunks.append(fallback_content)
            except:
                logger.error(f"Complete failure to process chunk {i}")
                continue
    
    logger.info(f"Extracted {len(text_chunks)} valid text chunks from {len(chunks)} input chunks")
    return text_chunks


def process_ocr_results(ocr_results: List[Any]) -> List[str]:
    """
    Process OCR results and extract text content.
    
    Args:
        ocr_results: List of OCRResult objects or strings
        
    Returns:
        List of strings containing OCR text
    """
    ocr_texts = []
    
    for i, result in enumerate(ocr_results):
        try:
            # If it's an OCRResult object, extract the text attribute
            if hasattr(result, 'text'):
                text_content = result.text.strip()
                if text_content and result.confidence > 10:  # Only accept results with some confidence
                    ocr_texts.append(text_content)
                    logger.debug(f"OCR result {i}: {len(text_content)} chars, confidence: {result.confidence:.1f}%")
                    
            # If it's already a string
            elif isinstance(result, str):
                text_content = result.strip()
                if text_content:
                    ocr_texts.append(text_content)
                    
            else:
                logger.warning(f"Unexpected OCR result type: {type(result)}")
                
        except Exception as e:
            logger.warning(f"Failed to process OCR result {i}: {e}")
            continue
    
    logger.info(f"Processed {len(ocr_texts)} valid OCR text chunks from {len(ocr_results)} results")
    return ocr_texts


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
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
            
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}",
            )

        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
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
        
        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as save_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {save_err}"
            )

        file_size_mb = get_size(save_path)
        logger.info(f"File saved at {save_path} ({file_size_mb} MB)")

        # OCR Processing (if PDF with images)
        ocr_texts = []
        if file_ext == ".pdf":
            try:
                logger.info("Starting OCR processing for PDF...")
                
                # Initialize OCR processor with fallback engines
                ocr_processor = AdvancedOCRProcessor(
                    primary_engine=OCREngine.EASYOCR,
                    fallback_engines=[OCREngine.PADDLEOCR, OCREngine.TESSERACT],
                    language=['en'],
                    gpu=True
                )
                
                # Extract images from PDF
                image_extractor = ExtractionImagesFromPDF(pdf_path=save_path)
                images = image_extractor.extract_images()
                
                if images:
                    logger.info(f"Extracted {len(images)} images from PDF")
                    
                    ocr_results = []
                    for i, img_path in enumerate(images):
                        try:
                            # Ensure full path
                            if not os.path.isabs(img_path):
                                full_img_path = os.path.join(MAIN_DIR, img_path)
                            else:
                                full_img_path = img_path
                                
                            if os.path.exists(full_img_path):
                                logger.debug(f"Processing image {i+1}/{len(images)}: {Path(full_img_path).name}")
                                
                                # Extract text from image
                                ocr_result = ocr_processor.extract_text(
                                    image=full_img_path,
                                    preprocess=True
                                )
                                
                                if ocr_result:
                                    ocr_results.append(ocr_result)
                                    
                            else:
                                logger.warning(f"Image file not found: {full_img_path}")
                                
                        except Exception as img_err:
                            logger.warning(f"Failed to process image {i+1}: {img_err}")
                            continue
                    
                    # Process OCR results to extract text
                    ocr_texts = process_ocr_results(ocr_results)
                    logger.info(f"OCR extracted {len(ocr_texts)} text chunks from {len(images)} images")
                    
                else:
                    logger.info("No images found in PDF for OCR processing")
                    
            except Exception as ocr_err:
                logger.warning(f"OCR extraction failed: {ocr_err}")
                ocr_texts = []  # Continue without OCR

        # Document Chunking
        try:
            logger.info("Starting document chunking...")
            chunks_meta = chunking_docs(file_path=save_path)
            
            if not chunks_meta or "chunks" not in chunks_meta:
                raise ValueError("Chunking failed - no chunks metadata returned")
                
            raw_chunks = chunks_meta.get("chunks", [])
            
            if not raw_chunks:
                raise ValueError("Chunking failed - no chunks produced")
                
            logger.info(f"Document chunking produced {len(raw_chunks)} raw chunks")
            
        except Exception as chunk_err:
            logger.error(f"Document chunking failed: {chunk_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to chunk document: {chunk_err}"
            )
        
        # Extract text content from chunks
        text_chunks = extract_text_from_chunks(raw_chunks)
        
        if not text_chunks and not ocr_texts:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text content could be extracted from the document"
            )

        # Combine all text chunks
        all_text_chunks = text_chunks + ocr_texts
        
        # Remove empty or very short chunks
        all_text_chunks = [chunk for chunk in all_text_chunks if chunk and len(chunk.strip()) > 10]
        
        if not all_text_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No meaningful text content found after processing"
            )
        
        logger.info(f"Total text chunks prepared for embedding: {len(all_text_chunks)} "
                   f"(document: {len(text_chunks)}, OCR: {len(ocr_texts)})")

        # Generate Embeddings
        try:
            logger.info("Generating embeddings...")
            embeddings_vectors = embedding_model.embed_texts(texts=all_text_chunks)
            
            if embeddings_vectors is None or len(embeddings_vectors) == 0:
                raise ValueError("Embedding model returned empty results")

            # Convert embeddings to numpy array if not already
            import numpy as np
            if not isinstance(embeddings_vectors, np.ndarray):
                if hasattr(embeddings_vectors, 'numpy'):
                    embeddings_vectors = embeddings_vectors.numpy()
                elif isinstance(embeddings_vectors, list):
                    if len(embeddings_vectors) > 0 and hasattr(embeddings_vectors[0], 'numpy'):
                        embeddings_vectors = np.array([emb.numpy() for emb in embeddings_vectors])
                    else:
                        embeddings_vectors = np.array(embeddings_vectors)
                else:
                    embeddings_vectors = np.array(embeddings_vectors)
            
            if embeddings_vectors.size == 0:
                raise ValueError("Generated embeddings array is empty")
                
            logger.info(f"Generated embedding array with shape: {embeddings_vectors.shape}")
            
        except Exception as embed_err:
            logger.error(f"Embedding generation failed: {embed_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate embeddings: {embed_err}"
            )

        # Build PathRAG Graph
        try:
            logger.info("Building PathRAG semantic graph...")
            
            graph_dir = os.path.join(MAIN_DIR, "pathrag_data")
            os.makedirs(graph_dir, exist_ok=True)
            graph_user_saved_dir = os.path.join(graph_dir, f"{user_id}")
            
            # Build the graph
            path_rag.build_graph(chunks=all_text_chunks, embeddings=embeddings_vectors, method="knn")

            # Save the graph
            autosave = AutoSave(pathrag_instance=path_rag, save_dir=graph_user_saved_dir)
            autosave.save_checkpoint()
            path_rag.save_graph(f"{graph_user_saved_dir}/{user_id}.pkl")
            
            # Verify the graph was saved
            if not os.path.exists(graph_user_saved_dir):
                raise ValueError("Graph file was not created successfully")
                
            logger.info(f"PathRAG graph built and saved to {graph_user_saved_dir}")

        except Exception as graph_err:
            logger.error(f"PathRAG graph building failed: {graph_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to build PathRAG graph: {graph_err}"
            )

        # Store metadata in MongoDB
        try:
            logger.info("Storing file metadata in MongoDB...")
            
            db = mongo_db[app_settings.MONGODB_NAME]
            collection = db["user_files"]
            
            metadata = {
                "user_id": user_id,
                "filename": unique_filename,
                "original_filename": file.filename,
                "file_path": save_path,
                "file_size_mb": file_size_mb,
                "file_extension": file_ext,
                "graph_path": graph_user_saved_dir,
                "num_chunks": len(all_text_chunks),
                "num_ocr_chunks": len(ocr_texts),
                "num_text_chunks": len(text_chunks),
                "embedding_dimension": embeddings_vectors.shape[1] if len(embeddings_vectors.shape) > 1 else None,
                "processing_timestamp": {"$currentDate": True}
            }
            
            # Remove existing entry for this user if exists
            collection.delete_many({"user_id": user_id, "filename": unique_filename})
            
            # Insert new metadata
            result = collection.insert_one(metadata)
            
            if result.inserted_id:
                logger.info(f"File metadata stored in MongoDB with ID: {result.inserted_id}")
            else:
                logger.warning("MongoDB insert may have failed - no ID returned")
                
        except Exception as db_err:
            logger.warning(f"MongoDB metadata storage failed: {db_err}")
            # Don't fail the entire request for database issues

        # Success Response
        response_data = {
            "status": "success",
            "message": "File processed successfully",
            "data": {
                "user_id": user_id,
                "file_name": unique_filename,
                "original_filename": file.filename,
                "file_size_mb": file_size_mb,
                "file_path": save_path,
                "graph_path": graph_user_saved_dir,
                "processing_stats": {
                    "total_chunks": len(all_text_chunks),
                    "document_chunks": len(text_chunks),
                    "ocr_chunks": len(ocr_texts),
                    "embedding_dimension": embeddings_vectors.shape[1] if len(embeddings_vectors.shape) > 1 else None
                }
            }
        }
        
        logger.info(f"File processing completed successfully for {file.filename}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected processing error: {str(e)}",
        )
