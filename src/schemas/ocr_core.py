

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OCREngine(Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    SURYA = "surya"


@dataclass
class OCRResult:
    """Standardized OCR result structure"""
    text: str
    confidence: float
    bbox: Optional[List[Tuple[int, int, int, int]]] = None
    word_boxes: Optional[List[Dict]] = None
    processing_time: Optional[float] = None
    engine: Optional[str] = None
    error: Optional[str] = None
