# Application Logging Documentation

## Overview

The Application Logging module provides a centralized, reusable logging utility with enhanced features for development and production environments. It combines colorized console output for improved development experience with rotating file logging for production monitoring and debugging.

## Features

- **Centralized Logging**: Single point of configuration for application-wide logging
- **Color-Coded Output**: ANSI-colored console messages based on severity levels
- **Rotating File Logging**: Efficient disk space management with automatic log rotation
- **Named Loggers**: Modular traceability across different components and files
- **Prevention of Duplication**: Caching mechanism prevents redundant logger initialization
- **Flexible Configuration**: Customizable log levels, directories, and file names
- **Cross-Environment Support**: Works in development, testing, and production

## Installation Requirements

```python
# Standard library only - no additional installations required
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys
```

## Quick Start

```python
from src.infra.logger import setup_logging

# Initialize logger
logger = setup_logging(name="MY_MODULE")

# Use logger
logger.info("Application started successfully")
logger.warning("Configuration file not found, using defaults")
logger.error("Database connection failed")
logger.debug("Processing user request with ID: 12345")
```

## Color Scheme

The console output uses ANSI color codes for improved readability:

| Level   | Color  | ANSI Code | Use Case |
|---------|--------|-----------|----------|
| DEBUG   | Green  | `\033[92m` | Detailed technical information |
| INFO    | Blue   | `\033[94m` | General application flow |
| WARNING | Yellow | `\033[93m` | Unexpected but non-critical events |
| ERROR   | Red    | `\033[91m` | Error conditions and failures |

## File Logging Configuration

### Default Settings
- **Location**: `<project_root>/logs/app.log`
- **Maximum Size**: 50MB per file
- **Backup Count**: 5 rotating files
- **Total Storage**: Up to 250MB (50MB × 5 files)
- **Rotation**: Automatic when file size limit is reached

### File Naming Convention
```
logs/
├── app.log           # Current active log
├── app.log.1         # Most recent backup
├── app.log.2         # Second most recent
├── app.log.3         # Third most recent
├── app.log.4         # Fourth most recent
└── app.log.5         # Oldest backup (gets deleted on next rotation)
```

## API Reference

### setup_logging()

The main function for configuring and retrieving logger instances.

```python
def setup_logging(
    name: str = "app_logger",
    log_dir: str = f"{MAIN_DIR}/logs",
    log_file: str = "app.log",
    console_level: int = logging.DEBUG,
) -> logging.Logger
```

**Parameters:**
- `name` (str): Unique identifier for the logger. Default: `"app_logger"`
- `log_dir` (str): Directory path for storing log files. Default: `"{MAIN_DIR}/logs"`
- `log_file` (str): Name of the log file. Default: `"app.log"`
- `console_level` (int): Minimum level for console output. Default: `logging.DEBUG`

**Returns:**
- `logging.Logger`: Configured logger instance with console and file handlers

**Features:**
- **Caching**: Returns existing logger if already initialized with the same name
- **Directory Creation**: Automatically creates log directory if it doesn't exist
- **Handler Management**: Prevents duplicate handlers from being added

### ColoredFormatter

Custom formatter class that adds ANSI color codes to console output.

```python
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""
    def format(self, record):
        message = super().format(record)
        return f"{COLORS.get(record.levelname, '')}{message}{COLORS['END']}"
```

## Usage Examples

### Basic Usage

```python
from src.infra.logger import setup_logging

# Initialize logger for specific module
logger = setup_logging(name="USER_SERVICE")

# Log different types of messages
logger.debug("User authentication process started")
logger.info("User logged in successfully")
logger.warning("User attempted invalid operation")
logger.error("User authentication failed")
```

### Module-Specific Loggers

```python
# In user_service.py
user_logger = setup_logging(name="USER_SERVICE")
user_logger.info("User service initialized")

# In database.py
db_logger = setup_logging(name="DATABASE")
db_logger.info("Database connection established")

# In api.py
api_logger = setup_logging(name="API")
api_logger.info("API server started")
```

### Custom Configuration

```python
# Custom log directory and file
logger = setup_logging(
    name="CUSTOM_MODULE",
    log_dir="./custom_logs",
    log_file="module.log",
    console_level=logging.INFO  # Only show INFO and above in console
)

# Production configuration
prod_logger = setup_logging(
    name="PRODUCTION",
    log_dir="/var/log/myapp",
    log_file="production.log",
    console_level=logging.WARNING  # Minimal console output
)
```

### Structured Logging

```python
logger = setup_logging(name="STRUCTURED")

# Include context in log messages
user_id = "12345"
operation = "file_upload"

logger.info(f"Operation started - User: {user_id}, Operation: {operation}")
logger.debug(f"Processing file - Size: 1024KB, Type: PDF")
logger.error(f"Operation failed - User: {user_id}, Error: Permission denied")
```

## Integration Patterns

### Class-Based Integration

```python
class DocumentProcessor:
    def __init__(self):
        self.logger = setup_logging(name="DOC_PROCESSOR")
        self.logger.info("DocumentProcessor initialized")
    
    def process_document(self, doc_path):
        self.logger.debug(f"Starting document processing: {doc_path}")
        try:
            # Process document
            result = self._process(doc_path)
            self.logger.info(f"Document processed successfully: {doc_path}")
            return result
        except Exception as e:
            self.logger.error(f"Document processing failed: {doc_path} - {e}")
            raise
```

### Function-Based Integration

```python
def upload_file(file_path):
    logger = setup_logging(name="FILE_UPLOAD")
    
    logger.info(f"File upload started: {file_path}")
    
    try:
        # Upload logic here
        logger.debug("Validating file format")
        validate_file(file_path)
        
        logger.debug("Uploading to storage")
        upload_to_storage(file_path)
        
        logger.info(f"File upload completed: {file_path}")
        
    except ValueError as e:
        logger.warning(f"File validation failed: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"File upload failed: {file_path} - {e}")
        raise
```

### Context Manager Integration

```python
import contextlib
from src.infra.logger import setup_logging

@contextlib.contextmanager
def logged_operation(operation_name, module_name="OPERATION"):
    logger = setup_logging(name=module_name)
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield logger
        logger.info(f"Operation completed successfully: {operation_name}")
    except Exception as e:
        logger.error(f"Operation failed: {operation_name} - {e}")
        raise

# Usage
with logged_operation("data_processing", "DATA_MODULE") as logger:
    logger.debug("Loading data from database")
    data = load_data()
    logger.debug("Processing data")
    result = process_data(data)
```

## Advanced Configuration

### Environment-Specific Logging

```python
import os
from src.infra.logger import setup_logging

def get_logger_for_environment(name):
    """Configure logger based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return setup_logging(
            name=name,
            log_dir="/var/log/myapp",
            console_level=logging.WARNING
        )
    elif env == "testing":
        return setup_logging(
            name=name,
            log_dir="./test_logs",
            console_level=logging.ERROR
        )
    else:  # development
        return setup_logging(
            name=name,
            console_level=logging.DEBUG
        )
```

### Custom Log Formatting

```python
# Extend ColoredFormatter for custom formatting
class TimestampColoredFormatter(ColoredFormatter):
    def format(self, record):
        # Add milliseconds to timestamp
        record.msecs = int(record.msecs)
        message = super().format(record)
        return message

def setup_custom_logging(name):
    logger = setup_logging(name=name)
    
    # Replace formatter on console handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            custom_formatter = TimestampColoredFormatter(
                "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(custom_formatter)
    
    return logger
```

### Performance Monitoring Integration

```python
import time
from functools import wraps

def log_execution_time(logger_name="PERFORMANCE"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging(name=logger_name)
            start_time = time.time()
            
            logger.debug(f"Starting {func.__name__}")
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s - {e}")
                raise
        return wrapper
    return decorator

# Usage
@log_execution_time("DATABASE")
def complex_query():
    # Simulate complex operation
    time.sleep(2)
    return "query_result"
```

## Best Practices

### Logger Naming Conventions

```python
# Use descriptive, hierarchical names
logger = setup_logging("MODULE_NAME")           # Good
logger = setup_logging("USER_AUTHENTICATION")  # Good  
logger = setup_logging("DATABASE_CONNECTION")  # Good

# Avoid generic names
logger = setup_logging("app")                   # Avoid
logger = setup_logging("main")                  # Avoid
```

### Log Level Guidelines

```python
logger = setup_logging("GUIDELINES")

# DEBUG: Detailed information for diagnosing problems
logger.debug("SQL query: SELECT * FROM users WHERE id = %s", user_id)
logger.debug("Function called with parameters: %s", params)

# INFO: General information about program execution
logger.info("User authentication successful")
logger.info("File processing completed")

# WARNING: Something unexpected happened, but the application can continue
logger.warning("Configuration file not found, using defaults")
logger.warning("Deprecated API endpoint used")

# ERROR: A serious problem occurred, functionality affected
logger.error("Database connection failed")
logger.error("File processing failed: %s", error_message)
```

### Exception Logging

```python
logger = setup_logging("EXCEPTION_HANDLING")

try:
    risky_operation()
except SpecificException as e:
    # Log specific exceptions with context
    logger.warning(f"Specific issue occurred: {e}")
    handle_specific_case()
except Exception as e:
    # Log unexpected exceptions with full traceback
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Performance Considerations

```python
logger = setup_logging("PERFORMANCE")

# Good: Lazy evaluation using % formatting
logger.debug("Processing item %s of %s", current, total)

# Avoid: String concatenation for debug messages
# logger.debug("Processing item " + str(current) + " of " + str(total))

# Good: Conditional logging for expensive operations
if logger.isEnabledFor(logging.DEBUG):
    expensive_debug_info = generate_debug_info()
    logger.debug("Debug info: %s", expensive_debug_info)
```

## Testing and Debugging

### Testing Logging Functionality

```python
import logging
import io
from unittest.mock import patch
from src.infra.logger import setup_logging

def test_logging_output():
    """Test that logging produces expected output."""
    
    # Capture log output
    log_stream = io.StringIO()
    logger = setup_logging("TEST")
    
    # Add custom handler for testing
    test_handler = logging.StreamHandler(log_stream)
    test_handler.setLevel(logging.INFO)
    logger.addHandler(test_handler)
    
    # Test logging
    logger.info("Test message")
    
    # Verify output
    output = log_stream.getvalue()
    assert "Test message" in output
    assert "INFO" in output
```

### Debugging Logger Configuration

```python
def debug_logger_info(logger_name):
    """Print detailed information about a logger's configuration."""
    logger = setup_logging(logger_name)
    
    print(f"Logger: {logger.name}")
    print(f"Level: {logging.getLevelName(logger.level)}")
    print(f"Handlers: {len(logger.handlers)}")
    
    for i, handler in enumerate(logger.handlers):
        print(f"  Handler {i}: {type(handler).__name__}")
        print(f"    Level: {logging.getLevelName(handler.level)}")
        print(f"    Formatter: {type(handler.formatter).__name__}")
```

## Log Analysis and Monitoring

### Log File Analysis

```python
import re
from collections import Counter
from datetime import datetime

def analyze_log_file(log_path):
    """Analyze log file for patterns and statistics."""
    
    level_counts = Counter()
    error_patterns = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract log level
            level_match = re.search(r' - (\w+) - ', line)
            if level_match:
                level_counts[level_match.group(1)] += 1
            
            # Collect error patterns
            if ' - ERROR - ' in line:
                error_patterns.append(line.strip())
    
    return {
        'level_counts': dict(level_counts),
        'error_patterns': error_patterns[:10]  # Top 10 errors
    }
```

### Real-time Log Monitoring

```python
import time
import os

def tail_log_file(log_path, callback=None):
    """Monitor log file for new entries."""
    
    def default_callback(line):
        print(f"New log entry: {line.strip()}")
    
    if callback is None:
        callback = default_callback
    
    with open(log_path, 'r') as f:
        # Go to end of file
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if line:
                callback(line)
            else:
                time.sleep(0.1)  # Wait for new content

# Usage
# tail_log_file("./logs/app.log")
```

## Production Deployment

### Log Rotation Management

```python
# Monitor log rotation status
def check_log_rotation(log_dir="./logs"):
    """Check status of log rotation."""
    from pathlib import Path
    
    log_files = list(Path(log_dir).glob("*.log*"))
    
    for log_file in sorted(log_files):
        stat = log_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"{log_file.name}: {size_mb:.1f}MB, modified: {mod_time}")
```

### Log Aggregation

```python
# Example integration with external log aggregation
import json

class JSONLogger:
    """Logger that outputs structured JSON for log aggregation."""
    
    def __init__(self, name):
        self.logger = setup_logging(name)
        self.name = name
    
    def structured_log(self, level, message, **kwargs):
        """Log structured data as JSON."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'logger': self.name,
            'level': level,
            'message': message,
            'metadata': kwargs
        }
        
        # Log as JSON string
        getattr(self.logger, level.lower())(json.dumps(log_data))

# Usage
json_logger = JSONLogger("STRUCTURED_SERVICE")
json_logger.structured_log("info", "User action", user_id="12345", action="login")
```

## Common Issues and Solutions

### Issue: Colors not showing in production
**Solution**: Colors are automatically disabled in non-TTY environments
```python
# Force color output if needed
import sys
if not sys.stdout.isatty():
    # Colors won't show in redirected output (normal behavior)
    pass
```

### Issue: Log files growing too large
**Solution**: Adjust rotation settings
```python
# Custom rotation configuration
from logging.handlers import RotatingFileHandler

def setup_custom_rotation(name, max_mb=10, backup_count=10):
    logger = logging.getLogger(name)
    
    file_handler = RotatingFileHandler(
        "app.log", 
        maxBytes=max_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count
    )
    logger.addHandler(file_handler)
    return logger
```

### Issue: Duplicate log messages
**Solution**: Ensure logger initialization is properly cached
```python
# The module already handles this, but if you see duplicates:
logger = setup_logging("MODULE")  # Returns cached instance
# Don't call setup_logging multiple times with the same name
```

### Issue: Logs not appearing in files
**Solution**: Check file permissions and directory existence
```python
import os
from pathlib import Path

def verify_log_setup(log_dir="./logs"):
    """Verify log directory setup."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory doesn't exist: {log_path}")
        return False
    
    if not os.access(log_path, os.W_OK):
        print(f"No write permission to log directory: {log_path}")
        return False
    
    print(f"Log directory is properly configured: {log_path}")
    return True
```

## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0
