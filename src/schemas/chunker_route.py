

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, validator

class ProcessingMode(str, Enum):
    ALL = "all"
    OCR_ONLY = "ocr_only"
    NO_OCR = "no_ocr"

@dataclass
class ProcessingConfig:
    batch_size: int = 100
    max_workers: int = 4
    max_file_size_mb: int = 100
    supported_formats: Tuple[str, ...] = ('.pdf', '.txt', '.docx', '.doc', '.md')

class ChunkingRequest(BaseModel):
    file_path: Optional[str] = None
    dir_file: Optional[str] = None
    reset_table: bool = False
    mode: ProcessingMode = ProcessingMode.ALL
    batch_size: int = 100

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('batch_size must be between 1 and 1000')
        return v

    @validator('file_path')
    def validate_file_path(cls, v):
        if v and not Path(v).exists():
            raise ValueError(f'File does not exist: {v}')
        return v

class ChunkingResponse(BaseModel):
    success: bool
    message: str
    total_chunks: int
    processed_files: int
    errors: List[str] = []
    processing_time_seconds: float

