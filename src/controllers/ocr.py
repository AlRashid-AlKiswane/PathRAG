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
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import tempfile
import json
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

# Import OCR libraries with fallback handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.error("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.error("EasyOCR not available. Install with: pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.error("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.error("TrOCR not available. Install with: pip install transformers torch")

try:
    from surya.ocr import run_ocr # type: ignore
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor # type: ignore
    from surya.model.recognition.model import load_model as load_rec_model # type: ignore
    from surya.model.recognition.processor import load_processor as load_rec_processor # type: ignore
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False
    logging.error("Surya OCR not available. Install with: pip install surya-ocr")

# Setup main directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.getLogger(__name__).error("Failed to set main directory path: %s", e)
    sys.exit(1)

from src.schemas import OCRResult, OCREngine
from src.infra import setup_logging

# Setup minimal logging - suppress errors from failed engines
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logger = logging.getLogger("OCR-OPERATIONS")
logger.setLevel(logging.CRITICAL)  # Suppress all non-critical messages

class AdvancedOCRProcessor:
    """
    Advanced OCR Image Processing class with multiple state-of-the-art models
    """
    
    def __init__(self,
                 primary_engine: OCREngine = OCREngine.EASYOCR,
                 fallback_engines: List[OCREngine] = None,
                 language: Union[str, List[str]] = 'en',
                 gpu: bool = True
) -> None:
        """
        Initialize Advanced OCR processor
        
        Args:
            primary_engine: Primary OCR engine to use
            fallback_engines: Fallback engines if primary fails
            language: Language codes (e.g., 'en', ['en', 'ar'])
            gpu: Use GPU acceleration if available
        """
        self.primary_engine = primary_engine
        self.fallback_engines = fallback_engines or [OCREngine.TESSERACT]
        self.language = language if isinstance(language, list) else [language]
        self.gpu = gpu and torch.cuda.is_available() if TROCR_AVAILABLE else False
        
        # Initialize engines
        self.engines = {}
        self._initialize_engines()
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        init_progress = tqdm(['EasyOCR', 'PaddleOCR', 'TrOCR', 'Surya', 'Tesseract'], 
                           desc="Initializing OCR Engines", 
                           bar_format='{l_bar}\033[92m{bar}\033[0m| {n_fmt}/{total_fmt}',
                           leave=False)
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE and OCREngine.EASYOCR in [self.primary_engine] + self.fallback_engines:
            try:
                self.engines[OCREngine.EASYOCR] = easyocr.Reader(
                    self.language, 
                    gpu=self.gpu,
                    verbose=False
                )
                init_progress.write(f"✅ EasyOCR ready (GPU: {self.gpu})")
            except Exception as e:
                init_progress.write(f"❌ EasyOCR failed: {e}")
        init_progress.update(1)

        # Initialize PaddleOCR  
        if PADDLEOCR_AVAILABLE and OCREngine.PADDLEOCR in [self.primary_engine] + self.fallback_engines:
            try:
                lang_code = 'en' if 'en' in self.language else self.language[0]
                # Fixed: Removed deprecated parameters and improved compatibility
                paddle_kwargs = {
                    'lang': lang_code,
                    'use_angle_cls': False,  # Disable to avoid compatibility issues
                    'det_db_thresh': 0.3,
                    'det_db_box_thresh': 0.5
                }
                
                # Only add GPU parameter if CUDA is available and we want GPU
                if self.gpu and torch and torch.cuda.is_available():
                    # Note: PaddleOCR uses different GPU parameter names in different versions
                    try:
                        self.engines[OCREngine.PADDLEOCR] = PaddleOCR(**paddle_kwargs, use_gpu=True)
                    except TypeError:
                        # Fallback if use_gpu parameter doesn't exist
                        self.engines[OCREngine.PADDLEOCR] = PaddleOCR(**paddle_kwargs)
                else:
                    self.engines[OCREngine.PADDLEOCR] = PaddleOCR(**paddle_kwargs)
                    
                init_progress.write(f"✅ PaddleOCR ready (GPU: {self.gpu})")
            except Exception as e:
                init_progress.write(f"❌ PaddleOCR failed: {e}")
        init_progress.update(1)
        
        # Initialize TrOCR
        if TROCR_AVAILABLE and OCREngine.TROCR in [self.primary_engine] + self.fallback_engines:
            try:
                device = "cuda" if self.gpu else "cpu"
                # Fixed: Corrected model name
                model_name = 'microsoft/trocr-large-printed'
                self.engines[OCREngine.TROCR] = {
                    'processor': TrOCRProcessor.from_pretrained(model_name),
                    'model': VisionEncoderDecoderModel.from_pretrained(model_name).to(device),
                    'device': device
                }
                init_progress.write(f"✅ TrOCR ready (device: {device})")
            except Exception as e:
                init_progress.write(f"❌ TrOCR failed: {e}")
        init_progress.update(1)
        
        # Initialize Surya OCR
        if SURYA_AVAILABLE and OCREngine.SURYA in [self.primary_engine] + self.fallback_engines:
            try:
                self.engines[OCREngine.SURYA] = {
                    'det_model': load_det_model(),
                    'det_processor': load_det_processor(),
                    'rec_model': load_rec_model(),
                    'rec_processor': load_rec_processor()
                }
                init_progress.write("✅ Surya OCR ready")
            except Exception as e:
                init_progress.write(f"❌ Surya OCR failed: {e}")
        init_progress.update(1)
        
        # Initialize Tesseract (fallback)
        if TESSERACT_AVAILABLE and OCREngine.TESSERACT in [self.primary_engine] + self.fallback_engines:
            try:
                # Auto-detect tesseract
                self._detect_tesseract()
                # Test tesseract functionality
                test_result = pytesseract.get_tesseract_version()
                self.engines[OCREngine.TESSERACT] = True
                init_progress.write(f"✅ Tesseract ready (v{test_result})")
            except Exception as e:
                init_progress.write(f"❌ Tesseract failed: {e}")
                # Remove from fallback engines if it fails
                if OCREngine.TESSERACT in self.fallback_engines:
                    self.fallback_engines.remove(OCREngine.TESSERACT)
        init_progress.update(1)
        init_progress.close()
        
        # Check if we have at least one working engine
        if not self.engines:
            tqdm.write("❌ No OCR engines could be initialized!")
            raise RuntimeError("No OCR engines available. Please check your installation.")
    
    def _detect_tesseract(self):
        """Auto-detect tesseract installation and configure tessdata"""
        import subprocess
        
        # Find tesseract executable
        possible_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', '/bin/tesseract']
        tesseract_cmd = None
        
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                break
        
        # Try which command if not found
        if not tesseract_cmd:
            try:
                result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
                if result.returncode == 0:
                    tesseract_cmd = result.stdout.strip()
            except Exception:
                pass
        
        if not tesseract_cmd:
            raise RuntimeError("Tesseract not found")
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Configure tessdata directory for Arch Linux
        possible_tessdata_paths = [
            '/usr/share/tessdata',
            '/usr/share/tesseract-ocr/4.00/tessdata',
            '/usr/share/tesseract-ocr/tessdata',
            '/usr/local/share/tessdata',
            '/opt/homebrew/share/tessdata'  # For some installations
        ]
        
        tessdata_dir = None
        for path in possible_tessdata_paths:
            if os.path.exists(os.path.join(path, 'eng.traineddata')):
                tessdata_dir = path
                break
        
        if tessdata_dir:
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            tqdm.write(f"✅ Found tessdata at: {tessdata_dir}")
        else:
            # Try to find tessdata with locate command
            try:
                result = subprocess.run(['locate', 'eng.traineddata'], capture_output=True, text=True)
                if result.returncode == 0:
                    eng_data_path = result.stdout.strip().split('\n')[0]
                    tessdata_dir = os.path.dirname(eng_data_path)
                    os.environ['TESSDATA_PREFIX'] = tessdata_dir
                    tqdm.write(f"✅ Found tessdata via locate: {tessdata_dir}")
                else:
                    tqdm.write("⚠️  Could not find tessdata. Install with: sudo pacman -S tesseract-data-eng")
            except Exception:
                tqdm.write("⚠️  Could not auto-detect tessdata directory")
    
    def preprocess_image(self, 
                        image: Union[str, np.ndarray, Image.Image],
                        enhance: bool = True,
                        denoise: bool = True,
                        deskew: bool = True,
                        resize_factor: Optional[float] = None) -> np.ndarray:
        """Enhanced image preprocessing with better error handling"""
        
        # Load image
        try:
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
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Deskewing
        if deskew:
            try:
                gray = self._deskew_image(gray)
            except Exception as e:
                logger.warning(f"Deskewing failed: {e}")
        
        # Resize for better OCR
        try:
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
        except Exception as e:
            logger.warning(f"Resizing failed: {e}")
        
        # Advanced denoising
        if denoise:
            try:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                gray = cv2.fastNlMeansDenoising(gray)
            except Exception as e:
                logger.warning(f"Denoising failed: {e}")
        
        # Enhanced contrast
        if enhance:
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                # Gamma correction
                gamma = 1.2
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                gray = cv2.LUT(gray, table)
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}")
        
        # Adaptive thresholding
        try:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 3
            )
        except Exception as e:
            logger.warning(f"Thresholding failed: {e}")
        
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
        """Extract text using EasyOCR with better error handling"""
        import time
        start_time = time.time()
        
        try:
            # Add timeout and better error handling
            results = self.engines[OCREngine.EASYOCR].readtext(
                image,
                width_ths=0.7,
                height_ths=0.7,
                paragraph=False
            )
            
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
        
        except KeyboardInterrupt:
            logger.info("EasyOCR processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
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
            logger.error(f"PaddleOCR extraction failed: {e}")
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
            logger.error(f"TrOCR extraction failed: {e}")
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
            logger.error(f"Surya OCR extraction failed: {e}")
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
            try:
                data = pytesseract.image_to_data(
                    pil_image,
                    lang='+'.join(self.language),
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except Exception:
                avg_confidence = 50.0 if text else 0.0
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                processing_time=time.time() - start_time,
                engine="Tesseract"
            )
        
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
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
        try:
            if preprocess:
                processed_img = self.preprocess_image(image)
            else:
                if isinstance(image, str):
                    processed_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                elif isinstance(image, Image.Image):
                    processed_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                else:
                    processed_img = image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return OCRResult(text="", confidence=0.0, error=f"Preprocessing failed: {e}")
        
        # Try primary engine
        result = None
        if target_engine in self.engines:
            try:
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
                if result and result.error is None and result.confidence > 30:
                    return result
            except KeyboardInterrupt:
                logger.info("OCR processing interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Primary engine {target_engine.value} failed: {e}")
        
        # Try fallback engines
        for fallback_engine in self.fallback_engines:
            if fallback_engine in self.engines and fallback_engine != target_engine:
                logger.info(f"Trying fallback engine: {fallback_engine.value}")
                
                try:
                    if fallback_engine == OCREngine.EASYOCR:
                        result = self._extract_with_easyocr(processed_img)
                    elif fallback_engine == OCREngine.PADDLEOCR:
                        result = self._extract_with_paddleocr(processed_img)
                    elif fallback_engine == OCREngine.TESSERACT:
                        result = self._extract_with_tesseract(processed_img)
                    elif fallback_engine == OCREngine.TROCR:
                        result = self._extract_with_trocr(processed_img)
                    elif fallback_engine == OCREngine.SURYA:
                        result = self._extract_with_surya(processed_img)
                    
                    if result and result.error is None and result.confidence > 20:
                        return result
                except KeyboardInterrupt:
                    logger.info("Fallback OCR processing interrupted by user")
                    raise
                except Exception as e:
                    logger.error(f"Fallback engine {fallback_engine.value} failed: {e}")
        
        # Return best result or empty result
        return result if result else OCRResult(text="", confidence=0.0, error="No engines available")


# Example usage with tqdm progress bars
if __name__ == "__main__":
    try:
        # Initialize with EasyOCR only (working engine)
        print("Initializing OCR with EasyOCR only...")
        ocr = AdvancedOCRProcessor(
            primary_engine=OCREngine.EASYOCR,
            fallback_engines=[],  # No fallbacks to avoid errors
            language=['en'],
            gpu=True
        )
        
        # Process images
        image_folder = '/home/alrashida/Tessafold/PathRAG/extracted_images'
        image_paths = [
            str(p) for p in Path(image_folder).glob('*')
            if p.suffix.lower() in ocr.supported_formats
        ]
        
        if not image_paths:
            print("❌ No images found in the specified folder")
            sys.exit(1)
        
        print(f"📁 Found {len(image_paths)} images to process")
        
        results = []
        chunk = ""
        successful_extractions = 0
        total_chars = 0
        
        # Create main progress bar with green color
        with tqdm(image_paths, desc="🔍 Processing Images", 
                 bar_format='{l_bar}\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                 colour='green') as pbar:
            
            for img_path in pbar:
                try:
                    # Update progress bar description
                    filename = Path(img_path).name
                    pbar.set_description(f"🔍 Processing: {filename[:20]}...")
                    
                    result = ocr.extract_text(img_path, preprocess=True)
                    
                    if result.text and len(result.text.strip()) > 0:
                        chunk += result.text + " "
                        results.append(result)
                        successful_extractions += 1
                        total_chars += len(result.text)
                        
                        # Update with success info
                        pbar.set_postfix({
                            'Success': f'{successful_extractions}/{len(image_paths)}',
                            'Chars': total_chars,
                            'Engine': result.engine,
                            'Conf': f'{result.confidence:.1f}%'
                        })
                    else:
                        pbar.set_postfix({
                            'Success': f'{successful_extractions}/{len(image_paths)}',
                            'Status': 'No text',
                            'Chars': total_chars
                        })
                        
                except KeyboardInterrupt:
                    pbar.write("⚠️  Processing interrupted by user")
                    break
                except Exception as e:
                    pbar.write(f"❌ Failed to process {filename}: {e}")
                    pbar.set_postfix({
                        'Success': f'{successful_extractions}/{len(image_paths)}',
                        'Status': 'Error',
                        'Chars': total_chars
                    })
                    continue
        
        # Final results summary
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        
        if chunk.strip():
            print(f"✅ Successfully processed: {successful_extractions}/{len(image_paths)} images")
            print(f"📝 Total extracted text: {len(chunk)} characters")
            print(f"📈 Average per successful image: {len(chunk)//max(successful_extractions, 1)} chars")
            
            # Show engines used
            engines_used = {}
            for result in results:
                engines_used[result.engine] = engines_used.get(result.engine, 0) + 1
            
            print(f"🔧 Engines used: {', '.join([f'{k}({v})' for k, v in engines_used.items()])}")
            
            print("\n💾 Text preview (first 300 chars):")
            print("-" * 40)
            preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
            print(preview)
            print("-" * 40)
        else:
            print("❌ No text was extracted from any images")
            print("💡 Suggestions:")
            print("   • Check if images contain readable text")
            print("   • Try installing tesseract language data: sudo pacman -S tesseract-data-eng")
            print("   • Consider image quality - low resolution may affect OCR")
            
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Program failed: {e}")
        sys.exit(1)
