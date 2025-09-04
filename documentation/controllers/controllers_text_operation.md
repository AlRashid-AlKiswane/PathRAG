# TextCleaner Documentation

## Overview

The `TextCleaner` class is a robust text preprocessing utility designed specifically for NLP applications and RAG (Retrieval-Augmented Generation) pipelines. It provides comprehensive text cleaning capabilities to normalize and sanitize raw text data for downstream processing.

## Features

- **Unicode Normalization**: Standardizes Unicode characters using NFKC normalization
- **HTML Tag Removal**: Strips HTML markup using BeautifulSoup
- **URL Removal**: Removes web URLs and www links
- **File Path Removal**: Removes Unix and Windows file paths
- **Citation Cleanup**: Removes bracketed citations and references (e.g., `[1]`, `[citation]`)
- **Non-ASCII Filtering**: Optional removal of emoji and special characters
- **Whitespace Normalization**: Collapses multiple spaces and removes newlines
- **Case Normalization**: Optional lowercase conversion

## Installation Requirements

```python
pip install beautifulsoup4
```

## Quick Start

```python
from text_cleaner import TextCleaner

# Initialize with default settings
cleaner = TextCleaner()

# Clean your text
raw_text = """
    Visit our site: https://example.com/docs.
    File saved at /home/user/project/file.txt
    <html><body><b>Example</b> content 😀</body></html>
    [1] Reference needed.
"""

cleaned_text = cleaner.clean(raw_text)
print(cleaned_text)
# Output: "Visit our site: . File saved at Example content ."
```

## Class Reference

### TextCleaner

A robust text cleaning utility for NLP preprocessing in RAG pipelines.

#### Constructor

```python
TextCleaner(lowercase=False, remove_html=True, remove_non_ascii=True)
```

**Parameters:**
- `lowercase` (bool, optional): Convert all text to lowercase. Default: `False`
- `remove_html` (bool, optional): Remove HTML tags using BeautifulSoup. Default: `True`
- `remove_non_ascii` (bool, optional): Remove characters outside ASCII range (emojis, special symbols). Default: `True`

#### Methods

##### clean(text: str) -> str

Clean the input text by applying all configured normalization and cleaning operations.

**Parameters:**
- `text` (str): The raw input text to be cleaned

**Returns:**
- `str`: Cleaned and normalized text

**Raises:**
- `ValueError`: If the input is not a string
- `RuntimeError`: If text cleaning fails due to unexpected error

## Cleaning Process

The `clean()` method applies the following operations in sequence:

1. **Unicode Normalization**: Applies NFKC normalization to standardize Unicode characters
2. **HTML Stripping**: Removes HTML tags if `remove_html=True`
3. **Case Conversion**: Converts to lowercase if `lowercase=True`
4. **URL Removal**: Removes HTTP/HTTPS URLs and www links using regex pattern `r'https?://\S+|www\.\S+'`
5. **File Path Removal**: Removes Unix and Windows file paths using pattern `r'([a-zA-Z]:)?(\/|\\)[\w\/\\\.\-]+'`
6. **Citation Removal**: Removes bracketed citations using pattern `r'\[\s*[^]]+\s*\]'`
7. **Line Break Removal**: Replaces newlines, carriage returns, and tabs with spaces
8. **Non-ASCII Removal**: Removes non-ASCII characters if `remove_non_ascii=True`
9. **Whitespace Normalization**: Collapses multiple whitespace characters into single spaces and trims

## Usage Examples

### Basic Usage

```python
# Default configuration
cleaner = TextCleaner()
result = cleaner.clean("Hello <b>World</b>! 😀")
# Output: "Hello World! "
```

### Custom Configuration

```python
# Keep case and non-ASCII characters
cleaner = TextCleaner(
    lowercase=False,
    remove_html=True,
    remove_non_ascii=False
)
result = cleaner.clean("Hello <b>World</b>! 😀")
# Output: "Hello World! 😀"
```

### RAG Pipeline Integration

```python
# Optimized for RAG preprocessing
cleaner = TextCleaner(
    lowercase=True,
    remove_html=True,
    remove_non_ascii=True
)

documents = [
    "Raw document with <tags>HTML</tags> and https://links.com",
    "Another doc with [citations] and /file/paths.txt"
]

cleaned_docs = [cleaner.clean(doc) for doc in documents]
```

## Error Handling

The TextCleaner includes comprehensive error handling:

- **Input Validation**: Raises `ValueError` for non-string inputs
- **Exception Wrapping**: Catches and re-raises unexpected errors as `RuntimeError`
- **Logging**: Uses structured logging for debugging and monitoring

```python
try:
    cleaner = TextCleaner()
    result = cleaner.clean(some_text)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Cleaning failed: {e}")
```

## Performance Considerations

- **BeautifulSoup**: HTML parsing adds overhead; disable with `remove_html=False` if not needed
- **Regex Operations**: Multiple regex operations are applied sequentially
- **Memory Usage**: Large texts are processed in-memory; consider chunking for very large documents

## Integration Notes

This module is designed to integrate with:
- **RAG Pipelines**: Clean documents before vectorization
- **NLP Preprocessing**: Standardize text for model input
- **Data Pipelines**: Batch processing of text data

The module follows the project's logging and configuration patterns, making it suitable for production environments.

## Logging

The TextCleaner uses structured logging with the logger name "TEXT-CLEANER". Log levels include:
- **INFO**: Initialization and configuration
- **ERROR**: Input validation failures
- **EXCEPTION**: Unexpected errors during processing

## Example Output

**Input:**
```
Visit our site: https://example.com/docs.
File saved at /home/user/project/file.txt
<html><body><b>Example</b> content 😀</body></html>
[1] Reference needed.
Multiple    spaces    and

newlines.
```

**Output (default settings):**
```
Visit our site: . File saved at Example content . Reference needed. Multiple spaces and newlines.
```

## Best Practices

1. **Configure Once**: Create TextCleaner instances with your desired configuration and reuse them
2. **Error Handling**: Always wrap cleaning operations in try-catch blocks
3. **Batch Processing**: For large datasets, consider processing in batches to manage memory
4. **Validation**: Validate input text before cleaning to avoid unnecessary processing
5. **Testing**: Test with representative data to ensure cleaning meets your requirements


## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0