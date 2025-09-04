# PDF Image Extraction Module

## Overview

The PDF Image Extraction Module provides robust functionality to extract images from PDF documents using PyMuPDF (fitz). This module processes all pages of a PDF file, identifies embedded images, and saves them to a specified output directory with comprehensive error handling and logging capabilities.

## Features

- **Complete PDF Processing**: Extracts images from all pages in a PDF document
- **Multiple Image Format Support**: Handles various embedded image formats (JPEG, PNG, etc.)
- **Robust Error Handling**: Page-level and image-level error recovery
- **Production-Ready Logging**: Comprehensive logging with structured output
- **Flexible Output Management**: Configurable output directory with automatic creation
- **Memory Efficient**: Processes images one at a time to minimize memory footprint

## Installation

### Prerequisites

```bash
pip install PyMuPDF  # Core PDF processing library
pip install Pillow   # Image format validation and processing
```

### Dependencies

- `PyMuPDF (fitz)`: PDF processing and image extraction
- `PIL (Pillow)`: Image format validation
- `os`, `sys`: File system operations
- Custom logging infrastructure (`src.infra`)

## Configuration

### System Requirements

- Python 3.7+
- Minimum 512MB RAM for small PDFs
- Disk space for extracted images (varies by PDF content)

## API Reference

### Class: `ExtractionImagesFromPDF`

Main class for extracting images from PDF files.

#### Constructor

```python
ExtractionImagesFromPDF(pdf_path: str, output_dir: str = "./extracted_images")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | `str` | Required | Path to the PDF file to extract images from |
| `output_dir` | `str` | `"./extracted_images"` | Directory path where extracted images will be saved |

#### Exceptions

| Exception | Description |
|-----------|-------------|
| `FileNotFoundError` | PDF file doesn't exist at specified path |
| `PermissionError` | Cannot access PDF or create output directory |

### Methods

#### `extract_images() -> List[str]`

Extract all images from the PDF and save them to the output directory.

**Returns:**
- `List[str]`: List of file paths for successfully extracted images

**Raises:**
- `Exception`: If PDF processing fails completely

**Image Naming Convention:**
- Format: `page{page_number}_img{image_number}.{extension}`
- Example: `page1_img1.jpeg`, `page2_img3.png`

## Usage Examples

### Basic Usage

```python
from pdf_image_extraction import ExtractionImagesFromPDF

# Extract images from a PDF
extractor = ExtractionImagesFromPDF("document.pdf", "output_images/")
extracted_paths = extractor.extract_images()

print(f"Extracted {len(extracted_paths)} images:")
for path in extracted_paths:
    print(f"  - {path}")
```

### Custom Output Directory

```python
import os
from datetime import datetime

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./extracted_images_{timestamp}"

extractor = ExtractionImagesFromPDF("report.pdf", output_dir)
images = extractor.extract_images()
```

### Batch Processing

```python
import os
from pathlib import Path

def batch_extract_images(pdf_directory: str, base_output_dir: str):
    """Extract images from all PDFs in a directory."""
    results = {}
    
    for pdf_file in Path(pdf_directory).glob("*.pdf"):
        try:
            # Create separate output directory for each PDF
            pdf_name = pdf_file.stem
            output_dir = os.path.join(base_output_dir, pdf_name)
            
            extractor = ExtractionImagesFromPDF(str(pdf_file), output_dir)
            extracted_paths = extractor.extract_images()
            
            results[str(pdf_file)] = {
                'status': 'success',
                'images_count': len(extracted_paths),
                'output_dir': output_dir
            }
            
            print(f"✓ {pdf_file.name}: {len(extracted_paths)} images extracted")
            
        except Exception as e:
            results[str(pdf_file)] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ {pdf_file.name}: {e}")
    
    return results

# Usage
results = batch_extract_images("./pdf_documents", "./extracted_images")
```

### Error Handling with Recovery

```python
import logging

def extract_with_recovery(pdf_path: str, output_dir: str, max_retries: int = 3):
    """Extract images with retry mechanism."""
    
    for attempt in range(max_retries):
        try:
            extractor = ExtractionImagesFromPDF(pdf_path, output_dir)
            images = extractor.extract_images()
            return images
            
        except FileNotFoundError:
            logging.error(f"PDF file not found: {pdf_path}")
            break  # No point retrying for missing file
            
        except PermissionError:
            logging.error(f"Permission denied accessing: {pdf_path}")
            break  # No point retrying for permission issues
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error(f"All {max_retries} attempts failed for {pdf_path}")
                raise
    
    return []

# Usage
images = extract_with_recovery("document.pdf", "./images", max_retries=3)
```

## Deployment Configuration

### Environment Setup

```bash
# Set logging level
export LOG_LEVEL=INFO

# Configure output permissions
export UMASK=0022  # Ensure readable extracted files
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create output directory with proper permissions
RUN mkdir -p /app/extracted_images && chmod 755 /app/extracted_images

# Set up volume for extracted images
VOLUME ["/app/extracted_images"]

CMD ["python", "-m", "src.pdf_image_extraction"]
```

### Production Considerations

1. **Storage Management**: Monitor disk space for extracted images
2. **Memory Usage**: Large PDFs may require memory optimization
3. **Concurrent Processing**: Module is thread-safe for parallel processing
4. **File Cleanup**: Implement cleanup strategies for temporary images
5. **Access Control**: Secure output directories appropriately

## Performance Metrics

| PDF Size | Pages | Typical Processing Time | Memory Usage | Output Size |
|----------|-------|------------------------|--------------|-------------|
| 1MB | 5-10 | 1-3 seconds | 50-100MB | 0.5-2MB |
| 10MB | 50-100 | 10-30 seconds | 100-300MB | 5-20MB |
| 50MB | 200+ | 1-5 minutes | 300-500MB | 20-100MB |

### Optimization Tips

```python
# For large PDFs, process in chunks
def extract_images_chunked(pdf_path: str, output_dir: str, chunk_size: int = 10):
    """Process PDF in page chunks to reduce memory usage."""
    extractor = ExtractionImagesFromPDF(pdf_path, output_dir)
    
    # Custom chunked processing implementation
    # (This would require modifying the extract_images method)
    pass
```

## Monitoring and Logging

### Log Levels and Messages

- **DEBUG**: Page processing details, image discovery
- **INFO**: Successful extractions, directory creation
- **WARNING**: Recoverable errors, skipped images
- **ERROR**: Image extraction failures, page processing errors
- **CRITICAL**: Complete processing failures

### Log Format

```
[TIMESTAMP] [LEVEL] [EX-Images-PDF] [MESSAGE]
```

### Monitoring Metrics

```python
# Key metrics to monitor in production
metrics = {
    'total_pdfs_processed': 0,
    'total_images_extracted': 0,
    'average_processing_time': 0.0,
    'error_rate': 0.0,
    'disk_space_used': '0MB'
}
```

## Testing

### Unit Test Example

```python
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pdf_image_extraction import ExtractionImagesFromPDF

class TestPDFImageExtraction(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = "test_document.pdf"
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('pdf_image_extraction.fitz.open')
    def test_successful_extraction(self, mock_fitz_open):
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        
        # Mock page with image
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(123, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'DCTDecode')]
        mock_doc.load_page.return_value = mock_page
        
        # Mock image extraction
        mock_doc.extract_image.return_value = {
            'image': b'fake_image_data',
            'ext': 'jpeg'
        }
        
        mock_fitz_open.return_value = mock_doc
        
        # Test extraction
        with patch('os.path.exists', return_value=True):
            extractor = ExtractionImagesFromPDF(self.test_pdf, self.temp_dir)
            
            with patch('builtins.open', mock_open()) as mock_file:
                images = extractor.extract_images()
                
                self.assertEqual(len(images), 1)
                self.assertIn('page1_img1.jpeg', images[0])

if __name__ == '__main__':
    unittest.main()
```

### Integration Test

```python
def test_real_pdf_extraction():
    """Integration test with actual PDF file."""
    import tempfile
    
    # Use a known test PDF
    test_pdf = "sample_document.pdf"  # Replace with actual test file
    temp_dir = tempfile.mkdtemp()
    
    try:
        extractor = ExtractionImagesFromPDF(test_pdf, temp_dir)
        images = extractor.extract_images()
        
        # Verify results
        assert len(images) > 0, "No images extracted from test PDF"
        
        for image_path in images:
            assert os.path.exists(image_path), f"Extracted image not found: {image_path}"
            assert os.path.getsize(image_path) > 0, f"Empty image file: {image_path}"
        
        print(f"✓ Successfully extracted {len(images)} images")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
```

## Troubleshooting

### Common Issues

1. **"PDF file not found"**
   - Verify file path is correct and accessible
   - Check file permissions

2. **"Permission denied"**
   - Ensure write permissions on output directory
   - Check disk space availability

3. **"No images found"**
   - PDF may contain vector graphics instead of embedded images
   - Some PDFs use text-based content only

4. **Memory errors with large PDFs**
   - Process PDFs in smaller chunks
   - Increase available system memory

### Debug Mode

```python
import logging

# Enable detailed logging
logging.getLogger('EX-Images-PDF').setLevel(logging.DEBUG)

# Extract with verbose output
extractor = ExtractionImagesFromPDF("document.pdf", "./images")
images = extractor.extract_images()
```

### Performance Debugging

```python
import time
import psutil
import os

def extract_with_profiling(pdf_path: str, output_dir: str):
    """Extract images with performance profiling."""
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    extractor = ExtractionImagesFromPDF(pdf_path, output_dir)
    images = extractor.extract_images()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Processing Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Usage: {end_memory - start_memory:.2f} MB")
    print(f"Images Extracted: {len(images)}")
    print(f"Average Time per Image: {(end_time - start_time) / max(1, len(images)):.3f} seconds")
    
    return images
```

## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0
