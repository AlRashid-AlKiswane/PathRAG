"""
Configuration Module for the Application.

This module defines the Settings class that centralizes application configuration.
Settings are automatically loaded from environment variables or a `.env` file
at runtime using `pydantic-settings`. Validation is built-in to ensure that
each configuration item adheres to expected types and value constraints.

Structure:
- Database configuration
- Document processing configuration
- Model identifiers for NER, embeddings, and generation
- Ollama LLM generation settings
- System monitoring thresholds

Usage:
    from app.core.config import get_settings
    settings = get_settings()

Raises:
    SystemExit: If required environment variables are missing or invalid.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Attempt to resolve the project root for imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logger.error("Failed to set up main directory path: %s", e)
    sys.exit(1)


class Settings(BaseSettings):
    """
    Centralized application settings loaded from environment variables.

    This class validates and exposes configuration variables for:
    - Database access
    - Document chunking and storage
    - Embedding and NER models
    - Ollama language model generation
    - System resource monitoring thresholds
    """

    # -----------------------------
    # Database Configuration
    # -----------------------------
    SQLITE_DB: str = Field(
        env="SQLITE_DB",
        description="SQLite database file path or URI."
    )

    # -----------------------------
    # Document Handling
    # -----------------------------
    FILE_TYPES: List[str] = Field(
        default=["text", "pdf"],
        env="FILE_TYPES",
        description="Allowed file types for upload and processing."
    )

    DOC_LOCATION_STORE: Path = Field(
        default=Path("./assets/docs"),
        env="DOC_LOCATION_STORE",
        description="Directory where uploaded documents will be stored."
    )

    CHUNKS_SIZE: int = Field(
        default=500,
        gt=100,
        lt=2000,
        env="CHUNKS_SIZE",
        description="Maximum number of characters in each text chunk."
    )

    CHUNKS_OVERLAP: int = Field(
        default=30,
        ge=0,
        lt=100,
        env="CHUNKS_OVERLAP",
        description="Number of overlapping characters between consecutive chunks."
    )

    # -----------------------------
    # Model Configuration
    # -----------------------------
    NER_MODEL: str = Field(
        default="dslim/distilbert-NER",
        env="NER_MODEL",
        description="HuggingFace model ID used for Named Entity Recognition."
    )

    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="Model name used for generating sentence embeddings."
    )

    EMBEDDING_DIM: int = Field(
        default=384,
        env="EMBEDDING_DIM",
        description="Embedding vector dimensionality (e.g., 384 for MiniLM)."
    )

    OLLAMA_EMBED_MODL: str = Field(
        default="all-minilm:l6-v2",
        env="OLLAMA_EMBED_MODL",
        description="Model used by Ollama for embedding generation."
    )

    OLLAMA_MODEL: str = Field(
        default="gemma3:1b",
        env="OLLAMA_MODEL",
        description="Ollama model name for large language model (LLM) inference."
    )

    # -----------------------------
    # Generation Parameters
    # -----------------------------
    MAX_NEW_TOKENS: int = Field(
        default=256,
        ge=10,
        lt=1024,
        env="MAX_NEW_TOKENS",
        description="Maximum number of new tokens the model is allowed to generate."
    )

    MAX_INPUT_TOKENS: int = Field(
        default=1024,
        ge=100,
        lt=4096,
        env="MAX_INPUT_TOKENS",
        description="Maximum number of tokens allowed in the input prompt."
    )

    TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        env="TEMPERATURE",
        description="Controls randomness in generation; higher values = more random."
    )

    TOP_K: int = Field(
        default=40,
        ge=1,
        le=100,
        env="TOP_K",
        description="Limits token sampling to the top-K most likely options."
    )

    TOP_P: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="TOP_P",
        description="Top-p (nucleus) sampling probability mass cutoff."
    )

    # -----------------------------
    # System Monitoring Thresholds
    # -----------------------------
    CPU_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="CPU_THRESHOLD",
        description="CPU usage threshold for triggering alerts or scaling."
    )

    MEMORY_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="MEMORY_THRESHOLD",
        description="Memory usage threshold for monitoring."
    )

    DISK_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="DISK_THRESHOLD",
        description="Disk usage threshold for storage alerts."
    )

    GPUs_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="GPUs_THRESHOLD",
        description="GPU usage threshold for scaling or warning."
    )

    MONITOR_INTERVAL: int = Field(
        default=60,
        ge=5,
        le=3600,
        env="MONITOR_INTERVAL",
        description="Interval (in seconds) between monitoring checks."
    )

    # -----------------------------
    # Pydantic Settings Metadata
    # -----------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


def get_settings() -> Settings:
    """
    Load and validate the application settings.

    Ensures that:
    - All required environment variables are loaded and validated.
    - The document storage directory exists or is created.

    Returns:
        Settings: A validated Settings instance with all configuration values.

    Raises:
        SystemExit: If validation fails or if file system setup fails.
    """
    try:
        settings = Settings()

        # Ensure the document directory exists
        settings.DOC_LOCATION_STORE.mkdir(parents=True, exist_ok=True)
        logger.info("Settings loaded successfully.")
        return settings

    except ValidationError as ve:
        logger.critical("Configuration validation failed:\n%s", ve.json(indent=2))
        sys.exit(1)

    except OSError as oe:
        logger.critical("Filesystem setup error: %s", oe)
        sys.exit(1)
