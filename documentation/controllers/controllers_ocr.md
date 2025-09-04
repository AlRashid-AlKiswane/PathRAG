# Advanced OCR Image Processing Module

## Overview

The Advanced OCR Image Processing Module provides comprehensive text extraction capabilities using multiple state-of-the-art OCR engines including Tesseract, EasyOCR, PaddleOCR, TrOCR, and Surya OCR. This module features intelligent fallback mechanisms, advanced image preprocessing, and robust error handling designed for production deployment on Arch Linux systems.

## Features

- **Multi-Engine Support**: 5 OCR engines with automatic fallback
- **Advanced Image Preprocessing**: Deskewing, denoising, contrast enhancement, and adaptive thresholding
- **GPU Acceleration**: CUDA support for compatible engines
- **Intelligent Fallback**: Automatic engine switching on failures
- **Progress Tracking**: Visual progress bars for batch operations
- **Production Logging**: Comprehensive error handling and monitoring
- **Format Support**: Multiple image formats (JPG, PNG, BMP, TIFF, WebP)
- **Language Support**: Multi-language OCR capabilities
- **Confidence Scoring**: Quality assessment for extracted text

## Installation

### Arch Linux System Dependencies

```bash
# Core OCR dependencies
sudo pacman -S tesseract tesseract-data-eng python-pip

# Image processing libraries
sudo pacman -S opencv python-pillow

# Optional: GPU drivers for CUDA acceleration
sudo pacman -S cuda cudnn
```

### Python Package Dependencies

```bash
# Basic OCR engines
pip install pytesseract easyocr

# Advanced engines
pip install paddlepaddle paddleocr
pip install transformers torch torchvision
pip install surya-ocr

# Image processing and utilities
pip install opencv-python pillow numpy requests tqdm

# GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Engine-Specific Requirements

| Engine | Size | GPU Support | Languages | Installation Command |
|--------|------|-------------|-----------|---------------------|
| Tesseract | Light | No | 100+ | `sudo pacman -S tesseract tesseract-data-eng` |
| EasyOCR | Medium | Yes | 80+ | `pip install easyocr` |
| PaddleOCR | Medium | Yes | 80+ | `pip install paddlepaddle paddleocr` |
| TrOCR | Large | Yes | English | `pip install transformers torch` |
| Surya | Large | Yes | 90+ | `pip install surya-ocr` |

## API Reference

### Class: `AdvancedOCRProcessor`

Main class for advanced OCR processing with multiple engine support.

#### Constructor

```python
AdvancedOCRProcessor(
    primary_engine: OCREngine = OCREngine.EASYOCR,
    fallback_engines: List[OCREngine] = None,
    language: Union[str, List[str]] = 'en',
    gpu: bool = True
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_engine` | `OCREngine` | `EASYOCR` | Primary OCR engine to use |
| `fallback_engines` | `List[OCREngine]` | `[TESSERACT]` | Fallback engines if primary fails |
| `language` | `Union[str, List[str]]` | `'en'` | Language codes (ISO 639-1) |
| `gpu` | `bool` | `True` | Enable GPU acceleration if available |

#### Engine Types

```python
from enum import Enum

class OCREngine(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    SURYA = "surya"
```

### Methods

#### `extract_text(image, preprocess=True, engine=None) -> OCRResult`

Extract text from an image using specified or primary OCR engine.

**Parameters:**
- `image`: Input image (str path, np.ndarray, or PIL.Image)
- `preprocess`: Apply image preprocessing (default: True)
- `engine`: Specific engine to use (None = use primary)

**Returns:**
```python
@dataclass
class OCRResult:
    text: str                           # Extracted text
    confidence: float                   # Average confidence score (0-100)
    word_boxes: List[Dict] = None      # Bounding boxes with text
    processing_time: float = 0.0       # Processing time in seconds
    engine: str = ""                   # Engine used for extraction
    error: str = None                  # Error message if failed
```

#### `preprocess_image(image, enhance=True, denoise=True, deskew=True, resize_factor=None) -> np.ndarray`

Advanced image preprocessing for improved OCR accuracy.

**Parameters:**
- `image`: Input image (various formats supported)
- `enhance`: Apply contrast enhancement and gamma correction
- `denoise`: Apply bilateral filtering and noise reduction
- `deskew`: Correct image skew using Hough transform
- `resize_factor`: Manual resize factor (None = auto-resize)

**Returns:**
- Preprocessed image as numpy array

## Usage Examples

### Basic Text Extraction

```python
from advanced_ocr_processor import AdvancedOCRProcessor
from src.schemas import OCREngine

# Initialize with default EasyOCR
ocr = AdvancedOCRProcessor()

# Extract text from image
result = ocr.extract_text("document.jpg")

print(f"Extracted Text: {result.text}")
print(f"Confidence: {result.confidence:.1f}%")
print(f"Engine Used: {result.engine}")
print(f"Processing Time: {result.processing_time:.2f}s")
```

### Multi-Engine Configuration

```python
# Configure with primary and fallback engines
ocr = AdvancedOCRProcessor(
    primary_engine=OCREngine.PADDLEOCR,
    fallback_engines=[OCREngine.EASYOCR, OCREngine.TESSERACT],
    language=['en', 'ar'],  # Multi-language support
    gpu=True
)

# Process with automatic fallback
result = ocr.extract_text("multilingual_document.png")

if result.error:
    print(f"Processing failed: {result.error}")
else:
    print(f"Success with {result.engine}: {result.text[:100]}...")
```

### Batch Processing with Progress Tracking

```python
from pathlib import Path
from tqdm import tqdm

def batch_ocr_processing(image_directory: str, output_file: str = "extracted_text.txt"):
    """Process multiple images with progress tracking."""
    
    ocr = AdvancedOCRProcessor(
        primary_engine=OCREngine.EASYOCR,
        fallback_engines=[OCREngine.TESSERACT],
        gpu=True
    )
    
    # Find supported image files
    image_paths = []
    for ext in ocr.supported_formats:
        image_paths.extend(Path(image_directory).glob(f"*{ext}"))
    
    if not image_paths:
        print("No supported images found")
        return
    
    results = []
    successful_extractions = 0
    total_text = ""
    
    # Process with progress bar
    with tqdm(image_paths, desc="Processing Images", unit="img") as pbar:
        for img_path in pbar:
            try:
                result = ocr.extract_text(str(img_path))
                
                if result.text and result.confidence > 30:
                    results.append(result)
                    total_text += f"\n--- {img_path.name} ---\n{result.text}\n"
                    successful_extractions += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Success': f'{successful_extractions}/{len(image_paths)}',
                    'Engine': result.engine,
                    'Conf': f'{result.confidence:.1f}%'
                })
                
            except KeyboardInterrupt:
                print("Processing interrupted by user")
                break
            except Exception as e:
                pbar.write(f"Failed {img_path.name}: {e}")
    
    # Save results
    if total_text:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(total_text)
        
        print(f"Extracted text from {successful_extractions} images")
        print(f"Total characters: {len(total_text)}")
        print(f"Saved to: {output_file}")
    else:
        print("No text was successfully extracted")
    
    return results

# Usage
results = batch_ocr_processing("./images", "extracted_text.txt")
```

### Custom Image Preprocessing

```python
import cv2
import numpy as np

def custom_preprocessing_pipeline(image_path: str):
    """Example of custom preprocessing pipeline."""
    
    ocr = AdvancedOCRProcessor()
    
    # Load image
    img = cv2.imread(image_path)
    
    # Custom preprocessing
    processed = ocr.preprocess_image(
        img,
        enhance=True,
        denoise=True,
        deskew=True,
        resize_factor=1.5  # Upscale by 50%
    )
    
    # Additional custom processing
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    # Extract text with custom preprocessing
    result = ocr.extract_text(processed, preprocess=False)
    
    return result

# Usage
result = custom_preprocessing_pipeline("low_quality_scan.jpg")
```

### Engine-Specific Processing

```python
def compare_all_engines(image_path: str):
    """Compare results across all available OCR engines."""
    
    # Initialize with all engines
    ocr = AdvancedOCRProcessor(gpu=True)
    
    engines_to_test = [
        OCREngine.TESSERACT,
        OCREngine.EASYOCR,
        OCREngine.PADDLEOCR,
        OCREngine.TROCR,
        OCREngine.SURYA
    ]
    
    results = {}
    
    for engine in engines_to_test:
        if engine in ocr.engines:  # Check if engine is available
            try:
                result = ocr.extract_text(image_path, engine=engine)
                results[engine.value] = {
                    'text': result.text,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'error': result.error
                }
            except Exception as e:
                results[engine.value] = {'error': str(e)}
    
    # Display comparison
    print("OCR Engine Comparison")
    print("=" * 50)
    
    for engine, result in results.items():
        print(f"\n{engine.upper()}:")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        else:
            print(f"  Confidence: {result['confidence']:.1f}%")
            print(f"  Time: {result['processing_time']:.2f}s")
            print(f"  Text: {result['text'][:100]}...")
    
    return results

# Usage
comparison = compare_all_engines("document.png")
```

## Configuration

### Environment Variables

```bash
# OCR engine preferences
export PRIMARY_OCR_ENGINE="easyocr"
export FALLBACK_OCR_ENGINES="tesseract,paddleocr"
export OCR_LANGUAGES="en,ar"

# GPU acceleration
export CUDA_VISIBLE_DEVICES=0
export OCR_USE_GPU=true

# Processing parameters
export OCR_CONFIDENCE_THRESHOLD=30.0
export IMAGE_PREPROCESS_ENABLED=true

# Tesseract configuration
export TESSDATA_PREFIX="/usr/share/tessdata"
export OMP_THREAD_LIMIT=4  # Limit threads for better performance
```

### Configuration Class

```python
from dataclasses import dataclass
from typing import List, Union

@dataclass
class OCRConfig:
    """Configuration class for OCR processing."""
    
    # Engine settings
    primary_engine: str = "easyocr"
    fallback_engines: List[str] = None
    languages: List[str] = None
    
    # Performance settings
    gpu_enabled: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30
    
    # Quality settings
    confidence_threshold: float = 30.0
    preprocess_enabled: bool = True
    deskew_enabled: bool = True
    denoise_enabled: bool = True
    
    # Image settings
    supported_formats: List[str] = None
    auto_resize_threshold: int = 600
    max_image_size: int = 4096
    
    def __post_init__(self):
        if self.fallback_engines is None:
            self.fallback_engines = ["tesseract"]
        if self.languages is None:
            self.languages = ["en"]
        if self.supported_formats is None:
            self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

# Usage
config = OCRConfig(
    primary_engine="paddleocr",
    fallback_engines=["easyocr", "tesseract"],
    languages=["en", "ar"],
    confidence_threshold=40.0
)
```

## Deployment Configuration

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchvision==0.14.1 \
    opencv-python-headless==4.7.1.72 \
    easyocr==1.7.0 \
    paddlepaddle==2.4.2 \
    paddleocr==2.6.1.3 \
    transformers==4.21.3 \
    pillow==9.5.0 \
    numpy==1.24.3 \
    tqdm==4.65.0

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directories for processing
RUN mkdir -p /app/data/input /app/data/output /app/logs
VOLUME ["/app/data", "/app/logs"]

# Set environment variables
ENV PYTHONPATH=/app
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV LOG_LEVEL=INFO
ENV OCR_USE_GPU=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.advanced_ocr_processor import AdvancedOCRProcessor; AdvancedOCRProcessor(gpu=False)"

CMD ["python", "-m", "src.advanced_ocr_processor"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-ocr-processor
  labels:
    app: ocr-processor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ocr-processor
  template:
    metadata:
      labels:
        app: ocr-processor
    spec:
      containers:
      - name: ocr-processor
        image: your-registry/advanced-ocr-processor:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: OCR_USE_GPU
          value: "false"
        - name: PRIMARY_OCR_ENGINE
          value: "easyocr"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: model-cache
          mountPath: /root/.cache
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from src.advanced_ocr_processor import AdvancedOCRProcessor; AdvancedOCRProcessor(gpu=False)"
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "print('healthy')"
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: ocr-data-pvc
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ocr-processor-service
spec:
  selector:
    app: ocr-processor
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### GPU-Enabled Deployment

```yaml
# For GPU acceleration in Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-processor-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ocr-processor-gpu
  template:
    metadata:
      labels:
        app: ocr-processor-gpu
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80  # Adjust based on your GPU
      containers:
      - name: ocr-processor
        image: your-registry/advanced-ocr-processor:gpu
        env:
        - name: OCR_USE_GPU
          value: "true"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4000m"
```

## Performance Metrics

### Benchmarks by Engine

| Engine | Speed (images/sec) | Accuracy | Memory Usage | GPU Support |
|--------|-------------------|----------|--------------|-------------|
| Tesseract | 2-5 | 85-90% | 100-200MB | No |
| EasyOCR | 1-3 | 88-93% | 500MB-1GB | Yes |
| PaddleOCR | 2-4 | 86-91% | 400-800MB | Yes |
| TrOCR | 0.5-1 | 90-95% | 1-2GB | Yes |
| Surya | 1-2 | 87-92% | 800MB-1.5GB | Yes |

### Performance Optimization

```python
import time
import psutil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    """Performance tracking for OCR operations."""
    
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    images_processed: int = 0
    successful_extractions: int = 0
    total_characters: int = 0
    average_confidence: float = 0.0
    
    def calculate_rates(self) -> Dict[str, float]:
        """Calculate processing rates."""
        if self.processing_time > 0:
            return {
                'images_per_second': self.images_processed / self.processing_time,
                'chars_per_second': self.total_characters / self.processing_time,
                'success_rate': self.successful_extractions / max(self.images_processed, 1) * 100
            }
        return {}

class PerformanceMonitor:
    """Monitor OCR performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = PerformanceMetrics()
        self.process = psutil.Process()
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
    
    def update(self, result: 'OCRResult'):
        """Update metrics with new result."""
        self.metrics.images_processed += 1
        
        if result.text and not result.error:
            self.metrics.successful_extractions += 1
            self.metrics.total_characters += len(result.text)
            
            # Update running average of confidence
            total_conf = (self.metrics.average_confidence * 
                         (self.metrics.successful_extractions - 1) + result.confidence)
            self.metrics.average_confidence = total_conf / self.metrics.successful_extractions
    
    def finish(self) -> PerformanceMetrics:
        """Finalize metrics calculation."""
        if self.start_time:
            self.metrics.processing_time = time.time() - self.start_time
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.metrics.memory_usage_mb = current_memory - self.initial_memory
        
        return self.metrics

# Usage example
def benchmark_ocr_engines(image_paths: list) -> Dict[str, PerformanceMetrics]:
    """Benchmark different OCR engines."""
    
    engines = [OCREngine.TESSERACT, OCREngine.EASYOCR, OCREngine.PADDLEOCR]
    results = {}
    
    for engine in engines:
        if not check_engine_availability(engine):
            continue
            
        print(f"Benchmarking {engine.value}...")
        
        ocr = AdvancedOCRProcessor(primary_engine=engine, fallback_engines=[])
        monitor = PerformanceMonitor()
        monitor.start()
        
        for img_path in tqdm(image_paths, desc=f"Testing {engine.value}"):
            try:
                result = ocr.extract_text(img_path)
                monitor.update(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        results[engine.value] = monitor.finish()
    
    return results
```

## Monitoring and Logging

### Production Logging Configuration

```python
import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime

class OCRLogFormatter(logging.Formatter):
    """Custom formatter for structured OCR logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add OCR-specific context if available
        if hasattr(record, 'engine'):
            log_entry['engine'] = record.engine
        if hasattr(record, 'confidence'):
            log_entry['confidence'] = record.confidence
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        
        return json.dumps(log_entry)

def setup_production_logging(log_dir: str = "/app/logs"):
    """Configure production logging for OCR processor."""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f"{log_dir}/ocr_processor.log",
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(OCRLogFormatter())
    
    # Error log handler
    error_handler = TimedRotatingFileHandler(
        f"{log_dir}/ocr_errors.log",
        when='midnight',
        interval=1,
        backupCount=30
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(OCRLogFormatter())
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # OCR-specific logger
    ocr_logger = logging.getLogger('OCR-OPERATIONS')
    ocr_logger.setLevel(logging.INFO)
    
    return ocr_logger

# Usage
logger = setup_production_logging()
```

### Health Check Endpoint

```python
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import asyncio

app = FastAPI(title="OCR Processor Health Check")

# Global OCR processor instance
ocr_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize OCR processor on startup."""
    global ocr_processor
    try:
        ocr_processor = AdvancedOCRProcessor(
            primary_engine=OCREngine.EASYOCR,
            fallback_engines=[OCREngine.TESSERACT],
            gpu=False
        )
        logger.info("OCR processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        raise

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engines": {},
        "memory_usage": {},
        "gpu_info": {}
    }
    
    try:
        # Check available engines
        for engine_name, engine_instance in ocr_processor.engines.items():
            health_status["engines"][engine_name.value] = {
                "available": True,
                "initialized": engine_instance is not None
            }
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        health_status["memory_usage"] = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
        
        # GPU info
        if torch.cuda.is_available():
            health_status["gpu_info"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024,
                "memory_cached": torch.cuda.memory_reserved() / 1024 / 1024
            }
        else:
            health_status["gpu_info"] = {"available": False}
        
        return health_status
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        raise HTTPException(status_code=500, detail=health_status)

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get processing metrics."""
    
    # This would typically connect to a metrics store
    # For now, return basic system metrics
    
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "num_threads": process.num_threads(),
        "create_time": process.create_time(),
        "engines_available": len(ocr_processor.engines) if ocr_processor else 0
    }
```

## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0
