# Unique Filename Generation Documentation

## Overview

The Unique Filename Generation module provides robust functionality to generate sanitized, unique filenames from original filename inputs. It's designed for applications that need to ensure file naming consistency, security, and uniqueness while maintaining traceability to the original filename.

## Features

- **Input Validation**: Validates filename strings and handles edge cases
- **File Type Verification**: Checks against allowed file extensions from application settings
- **Filename Sanitization**: Replaces special characters with safe alternatives
- **Uniqueness Guarantee**: Appends timestamp and UUID to prevent collisions
- **Fallback Mechanism**: Generates safe default filenames when errors occur
- **Comprehensive Logging**: Detailed logging of all processing steps and errors
- **Path Safety**: Handles various filename formats and edge cases

## Installation Requirements

```python
# Standard library modules (no additional installations required)
import os
import sys
import logging
import re
import uuid
from pathlib import Path
from datetime import datetime
```

## Quick Start

```python
from unique_filename import generate_unique_filename

# Generate a unique filename
original = "My Report 2025.pdf"
unique_name = generate_unique_filename(original)
print(unique_name)
# Output: "My_Report_2025_20250722_143052_a1b2c3d4.pdf"
```

## Function Reference

### generate_unique_filename(original_filename: str) -> str

Generate a unique, sanitized filename with timestamp and UUID suffix.

**Parameters:**
- `original_filename` (str): The original filename to process

**Returns:**
- `str`: A new sanitized filename with unique suffix, or `None` for unsupported file types

**Raises:**
- `ValueError`: If the input filename is invalid (empty, None, or not a string)

## Processing Steps

The function applies the following transformations in sequence:

### 1. Input Validation
- Checks if filename is a non-empty string
- Raises `ValueError` for invalid inputs

### 2. File Type Verification
- Validates extension against `app_settings.FILE_TYPES`
- Returns `None` for unsupported file types
- Logs warnings for rejected files

### 3. Component Extraction
- Uses `pathlib.Path` to safely extract name and extension
- Handles filenames with or without extensions
- Applies fallback values for missing components

### 4. Name Sanitization
- Replaces non-alphanumeric characters with underscores using regex `r"[^\w]"`
- Trims leading/trailing underscores
- Logs sanitization changes for debugging

### 5. Unique Suffix Generation
- Creates timestamp in format `YYYYMMDD_HHMMSS`
- Generates 8-character UUID hex string
- Combines as `timestamp_uuid`

### 6. Filename Assembly
- Constructs final filename: `{sanitized_name}_{unique_suffix}{extension}`
- Logs the transformation for audit trails

## Usage Examples

### Basic Usage

```python
# Standard document
result = generate_unique_filename("annual_report.pdf")
# Output: "annual_report_20250722_143052_a1b2c3d4.pdf"

# Filename with spaces and special characters
result = generate_unique_filename("User's Data (2025).xlsx")
# Output: "User_s_Data_2025_20250722_143052_b5c6d7e8.xlsx"
```

### Edge Cases

```python
# Missing extension
result = generate_unique_filename("document")
# Output: "document_20250722_143052_c7d8e9f0.dat"

# Empty name with extension
result = generate_unique_filename(".txt")
# Output: "file_20250722_143052_d8e9f0a1.txt"

# Unsupported file type
result = generate_unique_filename("script.exe")
# Output: None (if .exe not in FILE_TYPES)
```

### Error Handling

```python
try:
    result = generate_unique_filename("")  # Empty string
except ValueError as e:
    print(f"Invalid input: {e}")

try:
    result = generate_unique_filename(None)  # None input
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Configuration

The module requires application settings to define allowed file types:

```python
# In your settings configuration
FILE_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".png", ".jpg"]
```

## Fallback Mechanism

When unexpected errors occur during processing, the function provides a safe fallback:

```python
# Fallback filename format
"file_{timestamp}_{uuid}.dat"

# Example fallback
"file_20250722_143052_e9f0a1b2.dat"
```

## Filename Format

The generated filenames follow this pattern:

```
{sanitized_original_name}_{timestamp}_{short_uuid}{original_extension}
```

**Components:**
- **Sanitized Name**: Original filename with special characters replaced by underscores
- **Timestamp**: `YYYYMMDD_HHMMSS` format for chronological ordering
- **Short UUID**: 8-character hex string for uniqueness
- **Extension**: Original file extension (or `.dat` fallback)

## Security Considerations

### Filename Sanitization
- Removes potentially dangerous characters from filenames
- Prevents directory traversal attacks (e.g., `../../../etc/passwd`)
- Ensures compatibility across different operating systems

### Allowed File Types
- Validates against whitelist of permitted extensions
- Rejects potentially dangerous file types
- Configurable through application settings

## Performance Characteristics

### Time Complexity
- O(n) where n is the length of the filename
- Single-pass sanitization with regex
- Minimal overhead from UUID generation

### Memory Usage
- Low memory footprint
- String operations without large intermediate objects
- Suitable for high-volume file processing

## Integration Examples

### File Upload Handler

```python
def handle_file_upload(uploaded_file):
    """Process uploaded file with unique naming."""
    try:
        unique_name = generate_unique_filename(uploaded_file.name)
        if unique_name is None:
            return {"error": "Unsupported file type"}
        
        # Save file with unique name
        file_path = save_file(uploaded_file.data, unique_name)
        return {"success": True, "filename": unique_name, "path": file_path}
        
    except ValueError as e:
        return {"error": f"Invalid filename: {e}"}
```

### Batch File Processing

```python
def process_file_batch(filenames):
    """Generate unique names for multiple files."""
    results = []
    for original_name in filenames:
        try:
            unique_name = generate_unique_filename(original_name)
            results.append({
                "original": original_name,
                "unique": unique_name,
                "status": "success" if unique_name else "rejected"
            })
        except ValueError as e:
            results.append({
                "original": original_name,
                "unique": None,
                "status": "error",
                "message": str(e)
            })
    return results
```

## Logging

The module uses structured logging with the logger name "GET-UNIQUE-FILE-NAME":

### Log Levels
- **DEBUG**: Detailed processing information
- **WARNING**: Non-fatal issues (missing extensions, sanitization changes)
- **ERROR**: Input validation failures and processing errors

### Example Log Output
```
2025-07-22 14:30:52 [GET-UNIQUE-FILE-NAME] DEBUG: Sanitized filename from 'user's file' to 'user_s_file'
2025-07-22 14:30:52 [GET-UNIQUE-FILE-NAME] DEBUG: Generated new filename 'user_s_file_20250722_143052_a1b2c3d4.pdf' from original 'user's file.pdf'
2025-07-22 14:30:52 [GET-UNIQUE-FILE-NAME] WARNING: Skipping unsupported file type: malicious.exe
```

## Best Practices

### Input Validation
```python
# Always validate inputs before processing
if not filename or not isinstance(filename, str):
    raise ValueError("Invalid filename input")
```

### Error Handling
```python
# Handle both ValueError and fallback cases
try:
    unique_name = generate_unique_filename(original)
    if unique_name is None:
        # Handle unsupported file type
        handle_unsupported_file(original)
except ValueError as e:
    # Handle invalid input
    handle_invalid_input(e)
```

### File Type Configuration
```python
# Keep allowed file types restrictive and configurable
FILE_TYPES = [".pdf", ".docx", ".txt"]  # Only allow necessary types
```

### Logging Configuration
```python
# Configure appropriate log levels for production
import logging
logging.getLogger("GET-UNIQUE-FILE-NAME").setLevel(logging.WARNING)
```

## Testing Recommendations

### Unit Tests
- Test with valid filenames of different formats
- Test edge cases (empty names, missing extensions)
- Test invalid inputs (None, empty strings, non-strings)
- Test unsupported file types
- Verify uniqueness of generated names

### Integration Tests
- Test with actual file upload scenarios
- Verify file system compatibility
- Test batch processing performance
- Validate logging output

## Common Issues and Solutions

### Issue: Generated names too long
**Solution**: Truncate the original name before processing
```python
MAX_NAME_LENGTH = 50
truncated_name = original_name[:MAX_NAME_LENGTH] if len(original_name) > MAX_NAME_LENGTH else original_name
```

### Issue: Timestamp collisions in rapid processing
**Solution**: The UUID component provides sufficient uniqueness even for same-second operations

### Issue: Cross-platform filename compatibility
**Solution**: The sanitization process ensures compatibility across Windows, macOS, and Linux


## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0