"""
Application configuration settings loaded from environment variables.

This module provides a Settings class that loads configuration from:
1. .env file
"""

import sys
import logging
import os
from pathlib import Path
from typing import List
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)


class Settings(BaseSettings):
    """
    Application setting configuration with proper security measures.

    Note:
    - Sensitive credentials are stored as SecretStr
    - Required database paths are validated
    """

    # Database Configuration
    SQLITE_DB: str = Field(
        env="SQLITE_DB",
        description="SQLite database connection string"
    )

    # Document Processing
    FILE_TYPES: List[str] = Field(
        default=["text", "pdf"],
        env="FILE_TYPES",
        description="Supported file extensions for document processing"
    )

    DOC_LOCATION_STORE: Path = Field(
        default=Path("./assets/docs"),
        env="DOC_LOCATION_STORE",
        description="Directory to store uploaded documents"
    )

    CHUNKS_SIZE: int = Field(
        default=500,
        gt=100,
        lt=2000,
        env="CHUNKS_SIZE",
        description="Size of text chunks for processing (100 - 2000 chars)"
    )

    CHUNKS_OVERLAP: int = Field(
        default=30,
        ge=0,
        lt=100,
        env="CHUNKS_OVERLAP",
        description="Overlap between chunks (0-100 chars)"
    )

    # Model Configurations
    NER_MODEL: str = Field(
        default="dslim/distilbert-NER",
        env="NER_MODEL",
        description="Named Entity Recognition model identifier"
    )

    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="Sentence transformer model for embeddings"
    )

    EMBEDDING_DIM: int = Field(
        default=384,
        env="EMBEDDING_DIM",
        description="It maps senteces & pargraphs to a 384 dimensional dense vectore space."
    )

    OLLAMA_EMBED_MODL: str = Field(
        default="all-minilm:l6-v2",
        env="OLLAMA_EMBED_MODL",
        description=""
    )
    OLLAMA_MODEL: str = Field(
        default="gemma3:1b",
        env="OLLAMA_MODEL",
        description="LLM model used by Ollama"
    )

    # Ollama Configuration
    MAX_NEW_TOKENS: int = Field(
        default=256,
        ge=10,
        lt=1024,
        env="MAX_NEW_TOKENS",
        description="Maximum number of tokens to generate"
    )

    MAX_INPUT_TOKENS: int = Field(
        default=1024,
        ge=100,
        lt=4096,
        env="MAX_INPUT_TOKENS",
        description="Maximum number of input tokens allowed"
    )

    TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        env="TEMPERATURE",
        description="Sampling temperature for generation"
    )

    TOP_K: int = Field(
        default=40,
        ge=1,
        le=100,
        env="TOP_K",
        description="Top-K sampling for generation"
    )

    TOP_P: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        env="TOP_P",
        description="Top-P (nucleus) sampling for generation"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    """
    Safely initialize application settings with proper error handling.

    Returns:
        Settings: Configured settings instance

    Raises:
        SystemExit: If configuration is invalid
    """
    try:
        settings = Settings()

        # Ensure document directory exists
        # pylint: disable=no-member
        settings.DOC_LOCATION_STORE.mkdir(exist_ok=True, parents=True)

        return settings

    except ValidationError as e:
        print("Configuration error:", file=sys.stderr)
        print(e.json(indent=2), file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Filesystem error: {e}", file=sys.stderr)
        sys.exit(1)
