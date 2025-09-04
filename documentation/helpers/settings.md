# Application Configuration Documentation

## Overview

The Application Configuration module provides a robust, centralized settings management system for PathRAG applications. Built on Pydantic Settings, it offers environment-specific configurations, automatic validation, type safety, and seamless integration with `.env` files.

## Features

- **Environment-Specific Settings**: Separate configurations for development, production, and memory-constrained environments
- **Automatic Validation**: Type checking and value validation with Pydantic
- **Flexible Loading**: Support for custom `.env` files and environment variables
- **Directory Management**: Automatic creation of required directories
- **Robust Error Handling**: Graceful failure handling with detailed logging
- **Type Safety**: Full type hints and IDE support
- **Extensible Design**: Easy to add new configuration sections

## Installation Requirements

```python
pip install pydantic pydantic-settings
```

## Quick Start

```python
from settings import get_settings

# Load default configuration
settings = get_settings()

# Access configuration values
print(f"MongoDB URI: {settings.MONGODB_LOCAL_URI}")
print(f"Upload path: {settings.UPLOAD_STORAGE_PATH}")
print(f"Allowed types: {settings.UPLOAD_ALLOWED_TYPES}")

# Load from custom .env file
custom_settings = get_settings(env_file="production.env")
```

## Configuration Sections

### Development Environment
Optimized for local development with debugging enabled and reasonable resource limits.

```python
# Core development settings
DEV_DEBUG_MODE: bool = True
DEV_MAX_WORKERS: int = 2
DEV_BATCH_SIZE: int = 500
DEV_CACHE_SIZE_MB: int = 100
DEV_MEMORY_LIMIT_GB: float = 2.0
DEV_ENABLE_PROFILING: bool = True

# Algorithm parameters
DEV_DECAY_RATE: float = 0.8
DEV_PRUNE_THRESH: float = 0.5
DEV_SIM_THRESHOLD: float = 0.7
```

### Production Environment
High-performance settings optimized for production workloads.

```python
# Production scaling
PROD_DEBUG_MODE: bool = False
PROD_MAX_WORKERS: int = 8
PROD_BATCH_SIZE: int = 2000
PROD_CACHE_SIZE_MB: int = 1000
PROD_MEMORY_LIMIT_GB: float = 16.0
PROD_ENABLE_PROFILING: bool = False

# Optimized algorithm parameters
PROD_DECAY_RATE: float = 0.9
PROD_PRUNE_THRESH: float = 0.6
PROD_SIM_THRESHOLD: float = 0.75
```

### Memory-Constrained Environment
Minimal resource usage for constrained environments.

```python
# Memory optimization
MEMORY_MAX_WORKERS: int = 2
MEMORY_BATCH_SIZE: int = 200
MEMORY_CACHE_SIZE_MB: int = 50
MEMORY_MEMORY_LIMIT_GB: float = 1.0
MEMORY_ENABLE_SWAP: bool = True

# Conservative algorithm parameters
MEMORY_DECAY_RATE: float = 0.7
MEMORY_PRUNE_THRESH: float = 0.4
MEMORY_SIM_THRESHOLD: float = 0.65
```

### Database Configuration

```python
# MongoDB settings
MONGODB_PROVIDER: str = "local"
MONGODB_LOCAL_URI: str = "mongodb://localhost:27017/pathrag_ncec"
MONGODB_REMOTE_URI: Optional[str] = None
MONGODB_ENABLE_WEB_UI: bool = True
MONGODB_NAME: str = "pathrag_database"
```

### Document Processing

```python
# Upload settings
UPLOAD_ALLOWED_TYPES: List[str] = [".txt", ".pdf"]
UPLOAD_MAX_FILE_SIZE_MB: int = 50
UPLOAD_STORAGE_PATH: Path = Path("./storage/documents")
UPLOAD_TEMP_PATH: Path = Path("./storage/temp")

# Text processing
TEXT_CHUNK_SIZE: int = 512
TEXT_CHUNK_OVERLAP: int = 50
TEXT_MAX_CHUNK_COUNT: int = 10000
```

### Embedding Configuration

```python
# Embedding model
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS: int = 384
EMBEDDING_BATCH_SIZE: int = 32
EMBEDDING_DEVICE: str = "cpu"

# Ollama integration
OLLAMA_HOST: Optional[str] = None
OLLAMA_EMBEDDING_MODEL: Optional[str] = None
OLLAMA_CHAT_MODEL: Optional[str] = None
```

### Graph and PathRAG Settings

```python
# Graph construction
GRAPH_SIMILARITY_THRESHOLD: float = 0.75
GRAPH_DECAY_RATE: float = 0.9
GRAPH_PRUNING_THRESHOLD: float = 0.3
PATH_MAX_DEPTH: int = 5
BUILD_GRAPH_METHOD: str = "similarity"

# KNN retrieval
KNN_K_NEIGHBORS: int = 50
KNN_SIMILARITY_THRESHOLD: float = 0.7
KNN_INDEX_TYPE: str = "faiss"
KNN_METRIC: str = "cosine"
```

## Class Reference

### Settings

The main configuration class that inherits from `BaseSettings`.

```python
class Settings(BaseSettings):
    """Central application settings. Loads from environment and .env file by default."""
```

#### Key Methods

- **Automatic Loading**: Inherits environment variable loading from Pydantic Settings
- **Validation**: Built-in field validation with custom validators
- **Type Conversion**: Automatic type conversion for environment variables

### get_settings(env_file: Optional[str] = None) -> Settings

Load and validate application settings with directory creation and error handling.

**Parameters:**
- `env_file` (Optional[str]): Path to custom `.env` file. Defaults to `.env` in current directory.

**Returns:**
- `Settings`: Validated settings instance with all directories created

**Raises:**
- `SystemExit`: On validation errors or critical failures

## Environment File Format

Create a `.env` file in your project root:

```env
# Development Environment
DEV_DEBUG_MODE=true
DEV_MAX_WORKERS=4
DEV_BATCH_SIZE=1000
DEV_CACHE_SIZE_MB=200

# Database
MONGODB_PROVIDER=local
MONGODB_LOCAL_URI=mongodb://localhost:27017/pathrag_dev
MONGODB_NAME=pathrag_development

# Upload Configuration
UPLOAD_ALLOWED_TYPES=.txt,.pdf,.docx,.md
UPLOAD_MAX_FILE_SIZE_MB=100
UPLOAD_STORAGE_PATH=./data/documents
UPLOAD_TEMP_PATH=./data/temp

# Embedding Settings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
EMBEDDING_DEVICE=cuda

# Graph Configuration
GRAPH_SIMILARITY_THRESHOLD=0.8
GRAPH_DECAY_RATE=0.85
BUILD_GRAPH_METHOD=hierarchical

# Monitoring
MONITOR_CPU_THRESHOLD=0.8
MONITOR_MEMORY_THRESHOLD=0.85
MONITOR_HEALTH_CHECK_INTERVAL_SEC=60

# Features
FEATURE_WEB_INTERFACE=true
OPENAI_API_KEY=sk-your-openai-key-here
```

## Usage Examples

### Basic Configuration Loading

```python
from settings import get_settings

# Load default settings
config = get_settings()

# Access configuration values
print(f"Debug mode: {config.DEV_DEBUG_MODE}")
print(f"Upload path: {config.UPLOAD_STORAGE_PATH}")
print(f"Allowed file types: {config.UPLOAD_ALLOWED_TYPES}")
```

### Environment-Specific Loading

```python
# Load development configuration
dev_config = get_settings(env_file=".env.development")

# Load production configuration
prod_config = get_settings(env_file=".env.production")

# Use different settings based on environment
import os
env = os.getenv("ENVIRONMENT", "development")
config_file = f".env.{env}"
settings = get_settings(env_file=config_file)
```

### Database Connection Setup

```python
def setup_database(settings: Settings):
    """Initialize database connection using settings."""
    if settings.MONGODB_PROVIDER == "local":
        uri = settings.MONGODB_LOCAL_URI
    else:
        uri = settings.MONGODB_REMOTE_URI
    
    # Connect to MongoDB
    client = MongoClient(uri)
    database = client[settings.MONGODB_NAME]
    return database
```

### File Upload Configuration

```python
def configure_uploads(settings: Settings):
    """Configure file upload handling."""
    upload_config = {
        "allowed_types": settings.UPLOAD_ALLOWED_TYPES,
        "max_size_mb": settings.UPLOAD_MAX_FILE_SIZE_MB,
        "storage_path": settings.UPLOAD_STORAGE_PATH,
        "temp_path": settings.UPLOAD_TEMP_PATH
    }
    return upload_config
```

## Field Validators

### Boolean Parsing
Handles string boolean values from environment variables:

```python
@field_validator("MONGODB_ENABLE_WEB_UI", mode="before")
def _v_bool(cls, v):
    """Parse boolean values from strings."""
    # Accepts: "true", "1", "yes", "on", "y", "t"
    # Rejects: "false", "0", "no", "off", "n", "f"
```

### List Parsing
Converts comma-separated strings to lists:

```python
@field_validator("UPLOAD_ALLOWED_TYPES", mode="before")
def _v_upload_allowed_types(cls, v):
    """Parse comma-separated file extensions."""
    # ".txt,.pdf,.docx" -> [".txt", ".pdf", ".docx"]
```

### Path Validation
Converts strings to Path objects:

```python
@field_validator("UPLOAD_STORAGE_PATH", mode="before")
def _v_path(cls, v):
    """Convert string paths to Path objects."""
    return Path(v) if v else v
```

## Directory Management

The configuration system automatically creates required directories:

```python
# Directories created on startup
- UPLOAD_STORAGE_PATH
- UPLOAD_TEMP_PATH  
- DOC_LOCATION_STORE
- CHECKPOINT_DIRECTORY
- LOG_FILE_PATH.parent
```

## Error Handling

### Validation Errors
```python
try:
    settings = get_settings()
except SystemExit:
    # Handle configuration validation failure
    print("Configuration validation failed. Check your .env file.")
```

### Custom Error Handling
```python
def load_config_safely():
    """Load configuration with custom error handling."""
    try:
        return get_settings()
    except Exception as e:
        # Log error and provide fallback
        logging.error(f"Config loading failed: {e}")
        return get_default_settings()
```

## Logging Integration

The module includes comprehensive logging:

```python
# Logger configuration
LOG = logging.getLogger("settings")
LOG.setLevel(logging.INFO)

# Log messages include:
- Directory creation status
- Configuration loading success/failure
- Validation errors with detailed information
- Critical errors before system exit
```

## Best Practices

### Environment File Organization

```python
# Organize .env files by environment
.env                    # Default/development
.env.production        # Production settings
.env.testing          # Test environment
.env.staging          # Staging environment
```

### Sensitive Data Handling

```python
# Never commit sensitive data to version control
# Use environment-specific files for secrets
.env.production.local  # Local production overrides (gitignored)
.env.local            # Local development overrides (gitignored)
```

### Configuration Validation

```python
def validate_configuration(settings: Settings):
    """Additional validation beyond Pydantic."""
    if settings.PROD_MAX_WORKERS > os.cpu_count():
        logging.warning("Worker count exceeds CPU cores")
    
    if settings.UPLOAD_MAX_FILE_SIZE_MB > 1000:
        logging.warning("Large file upload limit may cause issues")
```

### Dynamic Configuration

```python
class ConfigurationManager:
    """Manage dynamic configuration updates."""
    
    def __init__(self):
        self.settings = get_settings()
        self._callbacks = []
    
    def reload_config(self, env_file: str = None):
        """Reload configuration from file."""
        try:
            self.settings = get_settings(env_file)
            self._notify_callbacks()
        except Exception as e:
            logging.error(f"Config reload failed: {e}")
    
    def register_callback(self, callback):
        """Register callback for configuration changes."""
        self._callbacks.append(callback)
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from settings import get_settings

app = FastAPI()
settings = get_settings()

@app.on_event("startup")
async def startup_event():
    # Initialize services with settings
    await setup_database(settings)
    await setup_embeddings(settings)

@app.get("/config")
async def get_config():
    return {
        "debug_mode": settings.DEV_DEBUG_MODE,
        "max_workers": settings.DEV_MAX_WORKERS,
        "allowed_types": settings.UPLOAD_ALLOWED_TYPES
    }
```

### CLI Integration

```python
import click
from settings import get_settings

@click.command()
@click.option("--config", help="Configuration file path")
def main(config):
    """Run application with specified configuration."""
    settings = get_settings(env_file=config)
    
    # Use settings to configure application
    run_application(settings)
```

## Testing Configuration

### Test Settings

```python
def get_test_settings():
    """Get configuration for testing."""
    return get_settings(env_file=".env.test")

# In tests
@pytest.fixture
def settings():
    return get_test_settings()

def test_upload_configuration(settings):
    assert ".txt" in settings.UPLOAD_ALLOWED_TYPES
    assert settings.UPLOAD_MAX_FILE_SIZE_MB > 0
```

### Mocking Settings

```python
from unittest.mock import patch
import pytest

@patch('settings.get_settings')
def test_with_mocked_config(mock_get_settings):
    # Create mock settings
    mock_settings = Settings(
        MONGODB_LOCAL_URI="mongodb://test:27017/test",
        UPLOAD_MAX_FILE_SIZE_MB=10
    )
    mock_get_settings.return_value = mock_settings
    
    # Test code using mocked settings
    result = process_with_config()
    assert result is not None
```

## Common Configuration Patterns

### Multi-Environment Setup

```python
def get_environment_config():
    """Load configuration based on ENVIRONMENT variable."""
    env = os.getenv("ENVIRONMENT", "development")
    config_files = {
        "development": ".env.dev",
        "staging": ".env.staging", 
        "production": ".env.prod",
        "testing": ".env.test"
    }
    
    config_file = config_files.get(env, ".env")
    return get_settings(env_file=config_file)
```

### Configuration Inheritance

```python
class ExtendedSettings(Settings):
    """Extended settings with additional fields."""
    
    # Additional custom settings
    CUSTOM_FEATURE_ENABLED: bool = Field(False, env="CUSTOM_FEATURE_ENABLED")
    CUSTOM_API_ENDPOINT: str = Field("http://localhost:8000", env="CUSTOM_API_ENDPOINT")
    
    @field_validator("CUSTOM_API_ENDPOINT")
    def validate_endpoint(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("API endpoint must start with http:// or https://")
        return v
```

## Performance Considerations

### Lazy Loading
```python
from functools import lru_cache

@lru_cache()
def get_cached_settings():
    """Cache settings to avoid repeated loading."""
    return get_settings()
```

### Memory Usage
- Settings are loaded once and reused
- Path objects are lightweight
- Validation occurs only at startup
- No runtime overhead after initialization

## Migration Guide

### From Simple Config

```python
# Before: Simple dictionary configuration
config = {
    "database_url": "mongodb://localhost:27017/app",
    "upload_path": "./uploads",
    "debug": True
}

# After: Pydantic settings
settings = get_settings()
database_url = settings.MONGODB_LOCAL_URI
upload_path = settings.UPLOAD_STORAGE_PATH
debug = settings.DEV_DEBUG_MODE
```

### Environment Variable Migration

```python
# Before: Manual environment variable reading
import os
debug_mode = os.getenv("DEBUG", "false").lower() == "true"
max_workers = int(os.getenv("MAX_WORKERS", "2"))

# After: Automatic type conversion and validation
settings = get_settings()
debug_mode = settings.DEV_DEBUG_MODE
max_workers = settings.DEV_MAX_WORKERS
```


## Author Information

- **Author**: AlRashid AlKiswane
- **Created**: 24-Aug-2025
- **Module Version**: 1.0.0
