r"""
Document Chunking and OCR Processing API Module - Fixed Version
=============================================================

This module provides a comprehensive REST API for document processing, chunking, and OCR extraction.
It supports batch processing of multiple document formats with intelligent chunking strategies,
OCR text extraction from PDF images, and MongoDB integration for persistent storage.

Core Features:
    - Multi-format document processing (PDF, TXT)
    - Intelligent text chunking with configurable strategies
    - Advanced OCR processing for PDF image content
    - Asynchronous batch processing with controlled concurrency
    - MongoDB integration with collection management
    - Comprehensive error handling and logging
    - Performance monitoring and optimization

Processing Modes:
    - ALL: Complete document chunking + OCR for PDFs (default)
    - OCR_ONLY: Extract only OCR content from PDF images
    - NO_OCR: Document chunking only, skip OCR processing

Author: ALRashid AlKiswane
Version: 1.1.0 (Fixed)
Last Modified: 2025-08-31
"""

import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
import time

# Third-party imports
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from tqdm import tqdm

# === Project Configuration ===
def setup_project_path() -> str:
    """Set up the project path and add to sys.path if needed."""
    try:
        main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if main_dir not in sys.path:
            sys.path.append(main_dir)
        return main_dir
    except Exception as e:
        print(f"Failed to configure project path: {e}")
        sys.exit(1)

# Initialize project path
MAIN_DIR = setup_project_path()

# === Project Imports ===
try:
    from src import get_mongo_db
    from src.controllers import (
        AdvancedOCRProcessor,
        ExtractionImagesFromPDF,
        TextCleaner,
        chunking_docs,
    )
    from src.mongodb import clear_collection, insert_chunk_to_mongo
    from src.helpers import Settings, get_settings
    from src.infra import setup_logging
    from src.schemas import (
        ChunkingRequest,
        ChunkingResponse,
        ProcessingConfig,
        ProcessingMode,
    )
    
    # Initialize logger and settings
    logger = setup_logging(name="CHUNKING-DOCS")
    app_settings: Settings = get_settings()
    
except ImportError as e:
    # Create a fallback logger if setup_logging fails
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("CHUNKING-DOCS")
    logger.error(f"Failed to import required modules: {e}")
    raise
except Exception as e:
    print(f"Unexpected error during imports: {e}")
    sys.exit(1)


class ChunkingService:
    """
    Advanced Document Processing and Chunking Service - Fixed Version
    ===============================================================
    
    A comprehensive service class that orchestrates document processing workflows including
    text extraction, intelligent chunking, OCR processing, and MongoDB storage operations.
    Designed for high-performance batch processing with robust error handling and monitoring.
    
    FIXES APPLIED:
    - Fixed incomplete __init__ method
    - Fixed typos in method names (porcess -> process)
    - Added proper return statements
    - Fixed configuration property names
    - Improved error handling and resource management
    - Added proper async context management
    - Fixed database insertion batching
    - Added comprehensive logging and monitoring
    """
    
    def __init__(self, db: MongoClient, config: Optional[ProcessingConfig] = None):
        """
        Initialize the ChunkingService with database connection and configuration.
        
        Args:
            db: MongoDB client connection
            config: Optional processing configuration, uses defaults if not provided
        """
        self.db = db
        self.config = config or ProcessingConfig()
        self.text_cleaner = TextCleaner(lowercase=True)
        self.ocr_processor = AdvancedOCRProcessor()
        self.errors: List[str] = []
        self.batch_size: int = getattr(app_settings, 'BATCH_SIZE', 512)
        self.max_workers: int = getattr(app_settings, 'MAX_WORKERS', 
                                      getattr(app_settings, 'CENTER_MAX_WORKERS', 4))
        
        logger.info(f"ChunkingService initialized with batch_size={self.batch_size}, "
                   f"max_workers={self.max_workers}")
    
    async def reset_collection_if_needed(self, reset: bool) -> bool:
        """Reset the chunks collection if requested."""
        if not reset:
            logger.debug("Collection reset not requested")
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, clear_collection, self.db, "chunks"
            )
            logger.info("Successfully cleared chunks collection")
            return True
        except Exception as e:
            error_msg = f"Failed to clear chunks collection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
    
    def validate_file_size(self, file_path: Path) -> None:
        """Validate file size against configured limits."""
        try:
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file_path}")
                
            size_mb = file_path.stat().st_size / (1024 * 1024)
            max_size = getattr(app_settings, 'UPLOAD_MAX_FILE_SIZE_MB', 100)
            
            if size_mb > max_size:
                raise ValueError(
                    f"File size ({size_mb:.1f}MB) exceeds limit ({max_size}MB)"
                )
                
            logger.debug(f"File size validation passed: {size_mb:.2f}MB")
            
        except OSError as e:
            raise ValueError(f"Cannot access file {file_path}: {str(e)}")

    def get_supported_files(self, directory: str | Path) -> List[Path]:
        """Get list of supported files from directory with validation."""
        # Convert to Path if it's a string
        if isinstance(directory, str):
            directory = Path(directory)

        # Validate directory
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        # Get supported file types
        supported_types = getattr(app_settings, 'FILE_TYPES', {'.pdf', '.txt'})
        
        files = []
        try:
            file_list = list(directory.iterdir())
        except PermissionError as e:
            raise ValueError(f"Permission denied accessing directory {directory}: {e}")
        
        logger.info(f"Scanning {len(file_list)} items in {directory}")
        
        for file_path in tqdm(file_list, desc="Validating files", unit="file"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_types):
                try:
                    self.validate_file_size(file_path)
                    files.append(file_path)
                    logger.debug(f"Added file: {file_path}")
                except ValueError as e:
                    error_msg = f"Skipping {file_path}: {e}"
                    self.errors.append(error_msg)
                    logger.warning(error_msg)
        
        logger.info(f"Found {len(files)} supported files out of {len(file_list)} total")
        return files

    async def process_ocr_for_pdf(self, pdf_path: Path, data_name: str) -> int:
        """Extract OCR content from PDF images - with tqdm progress bar."""
        try:
            logger.info(f"Starting OCR processing for: {pdf_path}")
            
            # Extract images from PDF
            extractor = ExtractionImagesFromPDF(pdf_path=str(pdf_path))
            image_paths = await asyncio.get_event_loop().run_in_executor(
                None, extractor.extract_images
            )

            if not image_paths:
                logger.info(f"No images found in PDF: {pdf_path}")
                return 0
                
            logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")
            
            # Process images with controlled concurrency
            content_parts = []
            
            # Use a smaller thread pool for OCR to prevent resource exhaustion
            ocr_workers = min(self.max_workers, len(image_paths), 2)
            
            with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
                # Submit all OCR tasks
                future_to_path = {}
                for img_path in image_paths:
                    full_img_path = os.path.join(MAIN_DIR, img_path)
                    if os.path.exists(full_img_path):
                        future = executor.submit(self.ocr_processor.extract_text, full_img_path)
                        future_to_path[future] = img_path
                
                # tqdm progress bar
                progress = tqdm(
                    total=len(future_to_path),
                    desc="Processing OCR",
                    unit="img",
                    colour="green"  # green progress bar
                )
                
                # Process completed futures
                for future in asyncio.as_completed(
                    [asyncio.wrap_future(f) for f in future_to_path.keys()]
                ):
                    try:
                        result = await future
                        if hasattr(result, 'text') and result.text:
                            # Handle both string and list text results
                            if isinstance(result.text, list):
                                text_content = " ".join(result.text)
                            else:
                                text_content = str(result.text)
                            
                            if text_content.strip():
                                content_parts.append(text_content)
                                
                    except Exception as e:
                        img_path = future_to_path.get(
                            next(f for f in future_to_path.keys() if f == future), 
                            "unknown"
                        )
                        error_msg = f"OCR processing failed for image {img_path}: {str(e)}"
                        self.errors.append(error_msg)
                        logger.error(error_msg)
                    finally:
                        progress.update(1)  # move tqdm bar one step
                
                progress.close()

            if not content_parts:
                logger.warning(f"No OCR content extracted from {pdf_path}")
                return 0
                
            # Combine and clean OCR content
            combined_content = "\n\n".join(content_parts)
            cleaned_content = self.text_cleaner.clean(combined_content)
            
            if cleaned_content.strip():
                chunk_data = [{
                    'chunk': cleaned_content,
                    'file': str(pdf_path),
                    'data_name': data_name,
                    'doc_id': str(uuid.uuid4()),
                    'size': len(cleaned_content),
                    'source_type': 'ocr',
                    'created_at': time.time()
                }]
                
                await self.insert_chunk_batch(chunk_data)
                logger.info(f"Successfully processed OCR content: {len(cleaned_content)} characters")
                return 1
            else:
                logger.warning(f"OCR content was empty after cleaning for {pdf_path}")
                
        except Exception as e:
            error_msg = f"OCR processing failed for {pdf_path}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            
        return 0

    async def process_document_chunking(self, file_path: Path, data_name: str) -> int:
        """Process document chunking for a single file."""
        try:
            logger.info(f"Starting document chunking for: {file_path}")
            
            # Run chunking in executor to avoid blocking
            meta = await asyncio.get_event_loop().run_in_executor(
                None, chunking_docs, str(file_path)
            )

            if not meta or not meta.get("chunks"):
                logger.warning(f"No chunks generated for {file_path}")
                return 0

            # Prepare chunks for batch insertion
            chunk_data = []
            for i, chunk in enumerate(meta["chunks"]):
                try:
                    # Handle different chunk object types
                    if hasattr(chunk, 'page_content'):
                        content = chunk.page_content
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', chunk.get('text', ''))
                    else:
                        content = str(chunk)
                    
                    cleaned_text = self.text_cleaner.clean(content)
                    
                    if cleaned_text.strip():  # Only insert non-empty chunks
                        chunk_data.append({
                            'chunk': cleaned_text,
                            'file': str(file_path),
                            'data_name': data_name,
                            'doc_id': str(uuid.uuid4()),
                            'size': len(cleaned_text),
                            'source_type': 'document',
                            'chunk_index': i,
                            'created_at': time.time()
                        })
                        
                except Exception as e:
                    error_msg = f"Failed to process chunk {i} from {file_path}: {str(e)}"
                    self.errors.append(error_msg)
                    logger.warning(error_msg)
            
            if chunk_data:
                await self.insert_chunk_batch(chunk_data)
                logger.info(f"Successfully processed {len(chunk_data)} chunks from {file_path}")
            else:
                logger.warning(f"No valid chunks extracted from {file_path}")
                
            return len(chunk_data)
            
        except Exception as e:
            error_msg = f"Document chunking failed for {file_path}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return 0

    async def insert_chunk_batch(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert chunks in batches for better performance with improved error handling."""
        if not chunks:
            logger.warning("No chunks to insert")
            return
            
        logger.info(f"Inserting {len(chunks)} chunks in batches of {self.batch_size}")
        
        successful_inserts = 0
        failed_inserts = 0
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._insert_batch_sync, batch
                )
                successful_inserts += len(batch)
                logger.debug(f"Successfully inserted batch {i//self.batch_size + 1}")
                
            except Exception as e:
                failed_inserts += len(batch)
                error_msg = f"Failed to insert batch {i//self.batch_size + 1}: {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
                
                # Continue with remaining batches instead of raising
                continue
        
        logger.info(f"Batch insertion completed: {successful_inserts} successful, "
                   f"{failed_inserts} failed")
        
        if failed_inserts > 0 and successful_inserts == 0:
            raise Exception(f"All batch insertions failed. Total failed: {failed_inserts}")

    def _insert_batch_sync(self, batch: List[Dict[str, Any]]) -> None:
        """Synchronous batch insert helper with improved error handling."""
        successful = 0
        failed = 0
        
        for chunk_data in batch:
            try:
                insert_chunk_to_mongo(
                    db=self.db,
                    chunk=chunk_data['chunk'],
                    file=chunk_data['file'],
                    data_name=chunk_data['data_name'],
                    doc_id=chunk_data['doc_id'],
                    size=chunk_data['size']
                )
                successful += 1
                
            except PyMongoError as e:
                failed += 1
                error_msg = f"MongoDB insertion failed for chunk: {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
                
            except Exception as e:
                failed += 1
                error_msg = f"Unexpected error during chunk insertion: {str(e)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
        
        if failed > 0:
            logger.warning(f"Batch insertion: {successful} successful, {failed} failed")
        
        if successful == 0 and failed > 0:
            raise Exception(f"All insertions in batch failed: {failed} errors")

    async def process_files(
        self,
        files: List[Path],
        data_name: str,
        mode: ProcessingMode
    ) -> Tuple[int, int]:
        """Process multiple files based on the specified mode with improved concurrency."""
        if not files:
            logger.warning("No files provided for processing")
            return 0, 0
            
        total_chunks = 0
        processed_files = 0
        
        logger.info(f"Processing {len(files)} files in mode: {mode}")

        # Create semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_single_file(file_path: Path) -> int:
            """Process a single file with proper resource management."""
            async with semaphore:
                chunks_added = 0
                file_start_time = time.time()
                
                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    # Document chunking (unless OCR-only mode)
                    if mode != ProcessingMode.OCR_ONLY:
                        logger.debug(f"Starting document chunking for {file_path}")
                        doc_chunks = await self.process_document_chunking(file_path, data_name)
                        chunks_added += doc_chunks
                        logger.debug(f"Document chunking added {doc_chunks} chunks")

                    # OCR processing for PDFs (unless no-OCR mode)
                    if (mode != ProcessingMode.NO_OCR and 
                        file_path.suffix.lower() == '.pdf'):
                        logger.debug(f"Starting OCR processing for {file_path}")
                        ocr_chunks = await self.process_ocr_for_pdf(file_path, data_name)
                        chunks_added += ocr_chunks
                        logger.debug(f"OCR processing added {ocr_chunks} chunks")
                    
                    processing_time = time.time() - file_start_time
                    logger.info(f"Completed processing {file_path}: "
                               f"{chunks_added} chunks in {processing_time:.2f}s")
                    
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {str(e)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    return 0
                
                return chunks_added
        
        # Process all files concurrently with progress tracking
        logger.info(f"Starting concurrent processing with {self.max_workers} workers")
        
        tasks = [process_single_file(file_path) for file_path in files]
        
        try:
            # Use asyncio.gather with return_exceptions to handle individual failures
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Task failed for {files[i]}: {str(result)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                elif isinstance(result, int):
                    if result > 0:
                        processed_files += 1
                    total_chunks += result
                else:
                    logger.warning(f"Unexpected result type for {files[i]}: {type(result)}")
        
        except Exception as e:
            logger.error(f"Critical error during concurrent processing: {str(e)}")
            raise
        
        logger.info(f"Processing completed: {processed_files}/{len(files)} files, "
                   f"{total_chunks} total chunks")
        
        return total_chunks, processed_files

    @asynccontextmanager
    async def _database_transaction(self):
        """Context manager for database operations with proper cleanup."""
        try:
            yield self.db
        except Exception as e:
            logger.error(f"Database transaction error: {e}")
            raise
        finally:
            # Perform any necessary cleanup
            pass


# === API Router ===
chunking_router = APIRouter(
    prefix="/api/v1/chunk",
    tags=["Document Processing"],
    responses={
        404: {"description": "Resource not found"},
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    }
)

@chunking_router.post("/", response_model=ChunkingResponse)
async def process_documents(
    request: ChunkingRequest,  # Fixed: Removed Depends() which was incorrect
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: MongoClient = Depends(get_mongo_db)
) -> ChunkingResponse:
    """
    Process documents with comprehensive error handling and performance optimization.
    
    Supports three processing modes:
    - all: Complete document chunking + OCR for PDFs
    - ocr_only: Extract only OCR content from PDF images  
    - no_ocr: Document chunking only, skip OCR
    """
    start_time = time.time()
    
    logger.info(f"Received processing request: {request}")

    # Validate input
    if not request.file_path and not request.dir_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'dir_file' must be provided"  # Fixed typo
        )

    # Initialize service
    service = ChunkingService(db=db)

    try:
        # Reset collection if requested
        collection_reset = await service.reset_collection_if_needed(request.reset_table)
        if collection_reset:
            logger.info("Collection was reset before processing")

        # Determine files to process
        files_to_process: List[Path] = []
        data_name = ""
        
        if request.file_path:
            file_path = Path(request.file_path)
            
            # Validate file exists and is accessible
            if not file_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found: {request.file_path}"
                )
            
            service.validate_file_size(file_path)
            files_to_process = [file_path]
            data_name = request.dir_file or file_path.stem
            
        elif request.dir_file:
            # Construct directory path
            dir_path = Path(MAIN_DIR) / "assets" / "docs" / request.dir_file
            
            try:
                files_to_process = service.get_supported_files(dir_path)
                data_name = request.dir_file
                
                if not files_to_process:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No supported files found in directory: {dir_path}"
                    )
                    
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
        
        # Validate processing mode
        if not isinstance(request.mode, ProcessingMode):
            logger.warning(f"Invalid processing mode: {request.mode}, using default")
            processing_mode = ProcessingMode.ALL
        else:
            processing_mode = request.mode
            
        logger.info(f"Processing {len(files_to_process)} files with mode: {processing_mode}")
        
        # Process files
        total_chunks, processed_files = await service.process_files(
            files_to_process, data_name, processing_mode
        )

        processing_time = time.time() - start_time
        
        # Log completion with structured data
        completion_info = {
            "event": "Document processing completed",
            "total_chunks": total_chunks,
            "processed_files": processed_files,
            "total_files": len(files_to_process),
            "processing_time": round(processing_time, 2),
            "errors_count": len(service.errors),
            "mode": str(processing_mode),
            "data_name": data_name
        }
        logger.info(completion_info)

        # Determine success based on results
        success = processed_files > 0 or total_chunks > 0
        
        if not success and service.errors:
            # If no files processed and there are errors, it's likely a failure
            message = f"Processing failed: {len(service.errors)} errors occurred"
        elif service.errors:
            message = f"Processing completed with {len(service.errors)} warnings/errors"
        else:
            message = "Document processing completed successfully"

        return ChunkingResponse(
            success=success,
            message=message,
            total_chunks=total_chunks,
            processed_files=processed_files,
            errors=service.errors,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error during processing after {processing_time:.2f}s", 
                    exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@chunking_router.get("/config", response_model=Dict[str, Any])
async def get_processing_config():
    """Get current processing configuration with all relevant settings."""
    try:
        # Create a service instance to get current batch size
        temp_service = ChunkingService(db=None)  # type: ignore
        
        config = {
            "batch_size": temp_service.batch_size,
            "max_workers": temp_service.max_workers,
            "max_file_size_mb": getattr(app_settings, 'UPLOAD_MAX_FILE_SIZE_MB', 100),
            "supported_formats": list(getattr(app_settings, 'FILE_TYPES', {'.pdf', '.txt'})),
            "processing_modes": [mode.value for mode in ProcessingMode],
            "version": "1.1.0",
            "service_name": "chunking"
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )

# Additional utility endpoints
@chunking_router.get("/stats")
async def get_processing_stats(db: MongoClient = Depends(get_mongo_db)):
    """Get processing statistics from the database."""
    try:
        chunks_collection = db.chunks
        
        # Get basic statistics
        total_chunks = await asyncio.get_event_loop().run_in_executor(
            None, chunks_collection.count_documents, {}
        )
        
        # Get aggregation stats
        pipeline = [
            {
                "$group": {
                    "_id": "$data_name",
                    "chunk_count": {"$sum": 1},
                    "total_size": {"$sum": "$size"},
                    "avg_size": {"$avg": "$size"}
                }
            }
        ]
        
        stats_by_dataset = list(await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(chunks_collection.aggregate(pipeline))
        ))
        
        return {
            "total_chunks": total_chunks,
            "datasets": stats_by_dataset,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )
