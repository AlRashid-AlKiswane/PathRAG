# Document Chunking Module

## Overview

The Document Chunking Module provides robust functionality to load and chunk documents (PDF or text files) into smaller pieces suitable for downstream processing, such as embedding or retrieval-augmented generation (RAG). This module is designed for production deployment with comprehensive error handling, logging, and configuration management.

## Features

- **Multi-format Support**: Handles PDF and text files seamlessly
- **Configurable Chunking**: Customizable chunk sizes and overlap settings
- **Robust Error Handling**: Comprehensive exception management with detailed logging
- **Production Ready**: Built with deployment considerations and monitoring capabilities
- **LangChain Integration**: Leverages proven document processing libraries

## Installation

### Prerequisites

```bash
pip install langchain-community
pip install PyMuPDF  # For PDF processing
```

### Dependencies

- `langchain_community`: Document loaders and text processing
- `langchain`: Text splitting functionality
- Custom application infrastructure (`src.infra`, `src.helpers`)

## Configuration

The module reads configuration from application settings:

```python
# Required settings in your configuration
FILE_TYPES = [".pdf", ".txt"]  # Supported file extensions
TEXT_CHUNK_SIZE = 1000         # Maximum characters per chunk
TEXT_CHUNK_OVERLAP = 200       # Character overlap between chunks
```

## API Reference

### `chunking_docs(file_path: Optional[str] = None) -> Dict[str, Any]`

Load and chunk a document into smaller pieces using LangChain's RecursiveCharacterTextSplitter.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `str` | Yes | Path to the document (PDF or text file) |

#### Returns

```python
{
    'chunks': List[Document],      # List of chunked documents
    'total_chunks': int,           # Total number of chunks created
    'file_path': str              # Original file path
}
```

#### Exceptions

| Exception | Description |
|-----------|-------------|
| `FileNotFoundError` | The specified file path does not exist |
| `ValueError` | Unsupported file type or no file path provided |
| `RuntimeError` | Document loading or chunking process failed |

## Usage Examples

### Basic Usage

```python
from chunking_module import chunking_docs

# Process a PDF document
result = chunking_docs("/path/to/document.pdf")
print(f"Created {result['total_chunks']} chunks from {result['file_path']}")

# Access individual chunks
for i, chunk in enumerate(result['chunks']):
    print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
```

### Error Handling

```python
import logging

try:
    result = chunking_docs("/path/to/document.pdf")
    # Process successful result
    process_chunks(result['chunks'])
    
except FileNotFoundError:
    logging.error("Document file not found")
except ValueError as e:
    logging.error(f"Invalid input: {e}")
except RuntimeError as e:
    logging.error(f"Processing failed: {e}")
```

### Batch Processing

```python
import os
from pathlib import Path

def process_directory(directory_path: str):
    """Process all supported documents in a directory."""
    results = []
    
    for file_path in Path(directory_path).glob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            try:
                result = chunking_docs(str(file_path))
                results.append(result)
                print(f"✓ Processed: {file_path.name} ({result['total_chunks']} chunks)")
            except Exception as e:
                print(f"✗ Failed: {file_path.name} - {e}")
    
    return results
```

## Deployment Configuration

### Environment Setup

```bash
# Set up logging level
export LOG_LEVEL=INFO

# Configure chunk settings
export TEXT_CHUNK_SIZE=1000
export TEXT_CHUNK_OVERLAP=200
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "-m", "src.chunking_module"]
```

### Production Considerations

1. **Memory Management**: Large documents may require memory optimization
2. **Concurrency**: Consider implementing async processing for multiple files
3. **Monitoring**: Logs are structured for production monitoring systems
4. **Scaling**: Module is stateless and suitable for horizontal scaling

## Performance Metrics

| File Type | Typical Processing Speed | Memory Usage |
|-----------|-------------------------|--------------|
| PDF (10MB) | ~2-5 seconds | ~50-100MB |
| TXT (1MB) | ~0.5-1 seconds | ~10-20MB |

## Monitoring and Logging

The module provides comprehensive logging at multiple levels:

- **DEBUG**: Detailed processing information
- **INFO**: Successful operations and progress
- **WARNING**: Non-critical issues (unsupported file types)
- **ERROR**: Processing failures and exceptions

### Log Format

```
[TIMESTAMP] [LEVEL] [CHUNKING-DOCS-CORE] [MESSAGE]
```

## Testing

### Unit Test Example

```python
import unittest
from unittest.mock import patch, MagicMock
from chunking_module import chunking_docs

class TestChunkingDocs(unittest.TestCase):
    
    @patch('chunking_module.Path')
    @patch('chunking_module.PyMuPDFLoader')
    def test_pdf_chunking_success(self, mock_loader, mock_path):
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.suffix.lower.return_value = ".pdf"
        
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = [MagicMock()]
        
        # Test
        result = chunking_docs("test.pdf")
        
        # Assertions
        self.assertIn('chunks', result)
        self.assertIn('total_chunks', result)
        self.assertIn('file_path', result)
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure file paths are absolute and accessible
2. **Unsupported Format**: Check `FILE_TYPES` configuration
3. **Memory Issues**: Reduce `TEXT_CHUNK_SIZE` for large documents
4. **Import Errors**: Verify all dependencies are installed

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('CHUNKING-DOCS-CORE')
```

