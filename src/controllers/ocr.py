#!/usr/bin/env python3
"""
Advanced OCR Image Processing Class with Multiple Models for Arch Linux
Supports: Tesseract, EasyOCR, PaddleOCR, TrOCR, and Surya OCR

Installation on Arch Linux:
# Basic requirements
sudo pacman -S tesseract tesseract-data-eng python-pip

# Python packages for advanced OCR
pip install easyocr paddlepaddle paddleocr transformers torch torchvision
pip install pillow opencv-python numpy requests surya-ocr

# For GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import tempfile
import json
from dataclasses import dataclass
from enum import Enum

# Import OCR libraries with fallback handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("TrOCR not available. Install with: pip install transformers torch")

try:
    from surya.ocr import run_ocr
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False
    print("Surya OCR not available. Install with: pip install surya-ocr")


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


class AdvancedOCRProcessor:
    """
    Advanced OCR Image Processing class with multiple state-of-the-art models
    """
    
    def __init__(self,
                 primary_engine: OCREngine = OCREngine.EASYOCR,
                 fallback_engines: List[OCREngine] = None,
                 language: Union[str, List[str]] = 'en',
                 gpu: bool = True,
                 enable_logging: bool = True):
        """
        Initialize Advanced OCR processor
        
        Args:
            primary_engine: Primary OCR engine to use
            fallback_engines: Fallback engines if primary fails
            language: Language codes (e.g., 'en', ['en', 'ar'])
            gpu: Use GPU acceleration if available
            enable_logging: Enable logging output
        """
        self.primary_engine = primary_engine
        self.fallback_engines = fallback_engines or [OCREngine.TESSERACT]
        self.language = language if isinstance(language, list) else [language]
        self.gpu = gpu and torch.cuda.is_available()
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
        
        # Initialize engines
        self.engines = {}
        self._initialize_engines()
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE and OCREngine.EASYOCR in [self.primary_engine] + self.fallback_engines:
            try:
                self.engines[OCREngine.EASYOCR] = easyocr.Reader(
                    self.language, 
                    gpu=self.gpu,
                    verbose=False
                )
                self.logger.info(f"EasyOCR initialized with GPU: {self.gpu}")
            except Exception as e:
                self.logger.error(f"Failed to initialize EasyOCR: {e}")
        
        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE and OCREngine.PADDLEOCR in [self.primary_engine] + self.fallback_engines:
            try:
                lang_code = 'en' if 'en' in self.language else self.language[0]
                self.engines[OCREngine.PADDLEOCR] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang_code,
                    use_gpu=self.gpu,
                    show_log=False
                )
                self.logger.info(f"PaddleOCR initialized with GPU: {self.gpu}")
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")
        
        # Initialize TrOCR
        if TROCR_AVAILABLE and OCREngine.TROCR in [self.primary_engine] + self.fallback_engines:
            try:
                device = "cuda" if self.gpu else "cpu"
                self.engines[OCREngine.TROCR] = {
                    'processor': TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed'),
                    'model': VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device),
                    'device': device
                }
                self.logger.info(f"TrOCR initialized on device: {device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize TrOCR: {e}")
        
        # Initialize Surya OCR
        if SURYA_AVAILABLE and OCREngine.SURYA in [self.primary_engine] + self.fallback_engines:
            try:
                self.engines[OCREngine.SURYA] = {
                    'det_model': load_det_model(),
                    'det_processor': load_det_processor(),
                    'rec_model': load_rec_model(),
                    'rec_processor': load_rec_processor()
                }
                self.logger.info("Surya OCR initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Surya OCR: {e}")
        
        # Initialize Tesseract (fallback)
        if TESSERACT_AVAILABLE and OCREngine.TESSERACT in [self.primary_engine] + self.fallback_engines:
            try:
                # Auto-detect tesseract
                self._detect_tesseract()
                self.engines[OCREngine.TESSERACT] = True
                self.logger.info("Tesseract initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Tesseract: {e}")
    
    def _detect_tesseract(self):
        """Auto-detect tesseract installation"""
        possible_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', '/bin/tesseract']
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return
        
        # Try which command
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                pytesseract.pytesseract.tesseract_cmd = result.stdout.strip()
                return
        except Exception:
            pass
        
        raise RuntimeError("Tesseract not found")
    
    def preprocess_image(self, 
                        image: Union[str, np.ndarray, Image.Image],
                        enhance: bool = True,
                        denoise: bool = True,
                        deskew: bool = True,
                        resize_factor: Optional[float] = None) -> np.ndarray:
        """Enhanced image preprocessing"""
        
        # Load image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError("Image must be path string, PIL Image, or numpy array")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Deskewing
        if deskew:
            gray = self._deskew_image(gray)
        
        # Resize for better OCR
        if resize_factor:
            height, width = gray.shape
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif gray.shape[1] < 600:  # Auto-resize small images
            scale = 600 / gray.shape[1]
            new_width = int(gray.shape[1] * scale)
            new_height = int(gray.shape[0] * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Advanced denoising
        if denoise:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray = cv2.fastNlMeansDenoising(gray)
        
        # Enhanced contrast
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Gamma correction
            gamma = 1.2
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # Adaptive thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        
        return gray
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image using Hough transform"""
        try:
            # Find edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate most common angle
                angles = []
                for rho, theta in lines[:20]:  # Use top 20 lines
                    angle = np.degrees(theta) - 90
                    angles.append(angle)
                
                # Get median angle
                angle = np.median(angles)
                
                # Rotate image if significant skew
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass  # Return original if deskewing fails
        
        return image
    
    def _extract_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR"""
        import time
        start_time = time.time()
        
        try:
            results = self.engines[OCREngine.EASYOCR].readtext(image)
            
            text_parts = []
            confidences = []
            word_boxes = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    text_parts.append(text)
                    confidences.append(confidence)
                    word_boxes.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) * 100 if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                word_boxes=word_boxes,
                processing_time=time.time() - start_time,
                engine="EasyOCR"
            )
        
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="EasyOCR",
                error=str(e)
            )
    
    def _extract_with_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR"""
        import time
        start_time = time.time()
        
        try:
            results = self.engines[OCREngine.PADDLEOCR].ocr(image, cls=True)
            
            text_parts = []
            confidences = []
            word_boxes = []
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        if confidence > 0.5:  # Filter low confidence
                            text_parts.append(text)
                            confidences.append(confidence)
                            word_boxes.append({
                                'text': text,
                                'bbox': bbox,
                                'confidence': confidence
                            })
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) * 100 if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                word_boxes=word_boxes,
                processing_time=time.time() - start_time,
                engine="PaddleOCR"
            )
        
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="PaddleOCR",
                error=str(e)
            )
    
    def _extract_with_trocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using TrOCR"""
        import time
        start_time = time.time()
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image).convert('RGB')
            
            # Process with TrOCR
            processor = self.engines[OCREngine.TROCR]['processor']
            model = self.engines[OCREngine.TROCR]['model']
            device = self.engines[OCREngine.TROCR]['device']
            
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # TrOCR doesn't provide confidence scores, estimate based on text quality
            confidence = 85.0 if len(text.strip()) > 0 else 0.0
            
            return OCRResult(
                text=text,
                confidence=confidence,
                processing_time=time.time() - start_time,
                engine="TrOCR"
            )
        
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="TrOCR",
                error=str(e)
            )
    
    def _extract_with_surya(self, image: np.ndarray) -> OCRResult:
        """Extract text using Surya OCR"""
        import time
        start_time = time.time()
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Run Surya OCR
            predictions = run_ocr(
                [pil_image],
                [self.language[0]],  # Surya expects single language
                self.engines[OCREngine.SURYA]['det_model'],
                self.engines[OCREngine.SURYA]['det_processor'],
                self.engines[OCREngine.SURYA]['rec_model'],
                self.engines[OCREngine.SURYA]['rec_processor']
            )
            
            if predictions:
                pred = predictions[0]
                text_parts = []
                confidences = []
                word_boxes = []
                
                for text_line in pred.text_lines:
                    text_parts.append(text_line.text)
                    # Surya provides confidence per text line
                    conf = getattr(text_line, 'confidence', 0.8) * 100
                    confidences.append(conf)
                    word_boxes.append({
                        'text': text_line.text,
                        'bbox': text_line.bbox,
                        'confidence': conf
                    })
                
                full_text = ' '.join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                return OCRResult(
                    text=full_text,
                    confidence=avg_confidence,
                    word_boxes=word_boxes,
                    processing_time=time.time() - start_time,
                    engine="Surya"
                )
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="Surya"
            )
        
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="Surya",
                error=str(e)
            )
    
    def _extract_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract (fallback)"""
        import time
        start_time = time.time()
        
        try:
            pil_image = Image.fromarray(image)
            
            # Extract text
            text = pytesseract.image_to_string(
                pil_image, 
                lang='+'.join(self.language),
                config='--oem 3 --psm 6'
            ).strip()
            
            # Get confidence data
            data = pytesseract.image_to_data(
                pil_image,
                lang='+'.join(self.language),
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                processing_time=time.time() - start_time,
                engine="Tesseract"
            )
        
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine="Tesseract",
                error=str(e)
            )
    
    def extract_text(self,
                    image: Union[str, np.ndarray, Image.Image],
                    preprocess: bool = True,
                    engine: Optional[OCREngine] = None) -> OCRResult:
        """
        Extract text using specified or primary OCR engine
        
        Args:
            image: Input image
            preprocess: Apply preprocessing
            engine: Specific engine to use (None = use primary)
        
        Returns:
            OCRResult object with text and metadata
        """
        # Use specified engine or primary
        target_engine = engine or self.primary_engine
        
        # Preprocess image
        if preprocess:
            processed_img = self.preprocess_image(image)
        else:
            if isinstance(image, str):
                processed_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            elif isinstance(image, Image.Image):
                processed_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            else:
                processed_img = image
        
        # Try primary engine
        if target_engine in self.engines:
            if target_engine == OCREngine.EASYOCR:
                result = self._extract_with_easyocr(processed_img)
            elif target_engine == OCREngine.PADDLEOCR:
                result = self._extract_with_paddleocr(processed_img)
            elif target_engine == OCREngine.TROCR:
                result = self._extract_with_trocr(processed_img)
            elif target_engine == OCREngine.SURYA:
                result = self._extract_with_surya(processed_img)
            elif target_engine == OCREngine.TESSERACT:
                result = self._extract_with_tesseract(processed_img)
            
            # If primary engine succeeds, return result
            if result.error is None and result.confidence > 30:
                return result
        
        # Try fallback engines
        for fallback_engine in self.fallback_engines:
            if fallback_engine in self.engines and fallback_engine != target_engine:
                self.logger.info(f"Trying fallback engine: {fallback_engine.value}")
                
                if fallback_engine == OCREngine.EASYOCR:
                    result = self._extract_with_easyocr(processed_img)
                elif fallback_engine == OCREngine.PADDLEOCR:
                    result = self._extract_with_paddleocr(processed_img)
                elif fallback_engine == OCREngine.TESSERACT:
                    result = self._extract_with_tesseract(processed_img)
                
                if result.error is None and result.confidence > 20:
                    return result
        
        # Return best result or empty result
        return result if 'result' in locals() else OCRResult(text="", confidence=0.0, error="No engines available")


# Example usage
if __name__ == "__main__":
    # Initialize with EasyOCR as primary, PaddleOCR and Tesseract as fallbacks
    ocr = AdvancedOCRProcessor(
        primary_engine=OCREngine.EASYOCR,
        fallback_engines=[OCREngine.PADDLEOCR, OCREngine.TESSERACT],
        language=['en'],
        gpu=True
    )
    
    # Process images
    image_folder = '/home/alrashida/Tessafold/PathRAG/extracted_images'
    image_paths = [
        str(p) for p in Path(image_folder).glob('*')
        if p.suffix.lower() in ocr.supported_formats
    ]
    
    results = []
    chunk = ""
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        result = ocr.extract_text(img_path, preprocess=True)
        chunk += "".join(result.text,)

    print(chunk)
