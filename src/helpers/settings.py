"""
Fixed application configuration module that is compatible with the provided .env.

Key features:
- Environment variable names aligned to .env keys (MONGODB_*, UPLOAD_*, EMBEDDING_*, etc.)
- Safe parsing for booleans, lists and paths
- Ensures important directories exist
- Robust validation logging and graceful exit on fatal config errors
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import Field, ValidationError, field_validator, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG = logging.getLogger("settings")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on", "y", "t"):
        return True
    if v in ("0", "false", "no", "off", "n", "f"):
        return False
    return None


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    # split by comma, strip spaces, ignore empty
    return [p.strip() for p in str(value).split(",") if p.strip()]


class Settings(BaseSettings):
    """
    Central application settings. Loads from environment and .env file by default.
    """

    # Pydantic Settings meta: read .env and ignore unknown vars so big .env files don't fail.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Center
    CENTER_MAX_WORKERS: int = Field(2, env="DEV_MAX_WORKERS")

    # -----------------------
    # Development Environment
    # -----------------------
    DEV_DEBUG_MODE: bool = Field(True, env="DEV_DEBUG_MODE")
    DEV_MAX_WORKERS: int = Field(2, env="DEV_MAX_WORKERS")
    DEV_BATCH_SIZE: int = Field(500, env="DEV_BATCH_SIZE")
    DEV_CACHE_SIZE_MB: int = Field(100, env="DEV_CACHE_SIZE_MB")
    DEV_MEMORY_LIMIT_GB: float = Field(2.0, env="DEV_MEMORY_LIMIT_GB")
    DEV_MAX_GRAPH_NODES: int = Field(10000, env="DEV_MAX_GRAPH_NODES")
    DEV_CHECKPOINT_INTERVAL_DOCS: int = Field(1000, env="DEV_CHECKPOINT_INTERVAL_DOCS")
    DEV_ENABLE_PROFILING: bool = Field(True, env="DEV_ENABLE_PROFILING")
    
    # New required fields
    DEV_DECAY_RATE: float = Field(0.8, env="DEV_DECAY_RATE")
    DEV_PRUNE_THRESH: float = Field(0.5, env="DEV_PRUNE_THRESH")
    DEV_SIM_THRESHOLD: float = Field(0.7, env="DEV_SIM_THRESHOLD")

    # -----------------------
    # Production Environment
    # -----------------------
    PROD_DEBUG_MODE: bool = Field(False, env="PROD_DEBUG_MODE")
    PROD_MAX_WORKERS: int = Field(8, env="PROD_MAX_WORKERS")
    PROD_BATCH_SIZE: int = Field(2000, env="PROD_BATCH_SIZE")
    PROD_CACHE_SIZE_MB: int = Field(1000, env="PROD_CACHE_SIZE_MB")
    PROD_MEMORY_LIMIT_GB: float = Field(16.0, env="PROD_MEMORY_LIMIT_GB")
    PROD_MAX_GRAPH_NODES: int = Field(100000, env="PROD_MAX_GRAPH_NODES")
    PROD_CHECKPOINT_INTERVAL_DOCS: int = Field(5000, env="PROD_CHECKPOINT_INTERVAL_DOCS")
    PROD_ENABLE_PROFILING: bool = Field(False, env="PROD_ENABLE_PROFILING")

    PROD_DECAY_RATE: float = Field(0.9, env="PROD_DECAY_RATE")
    PROD_PRUNE_THRESH: float = Field(0.6, env="PROD_PRUNE_THRESH")
    PROD_SIM_THRESHOLD: float = Field(0.75, env="PROD_SIM_THRESHOLD")

    # -----------------------
    # Memory-Constrained Environment
    # -----------------------
    MEMORY_MAX_WORKERS: int = Field(2, env="MEMORY_MAX_WORKERS")
    MEMORY_BATCH_SIZE: int = Field(200, env="MEMORY_BATCH_SIZE")
    MEMORY_CACHE_SIZE_MB: int = Field(50, env="MEMORY_CACHE_SIZE_MB")
    MEMORY_MEMORY_LIMIT_GB: float = Field(1.0, env="MEMORY_MEMORY_LIMIT_GB")
    MEMORY_MAX_GRAPH_NODES: int = Field(5000, env="MEMORY_MAX_GRAPH_NODES")
    MEMORY_CHECKPOINT_INTERVAL_DOCS: int = Field(500, env="MEMORY_CHECKPOINT_INTERVAL_DOCS")
    MEMORY_ENABLE_SWAP: bool = Field(True, env="MEMORY_ENABLE_SWAP")

    MEMORY_DECAY_RATE: float = Field(0.7, env="MEMORY_DECAY_RATE")
    MEMORY_PRUNE_THRESH: float = Field(0.4, env="MEMORY_PRUNE_THRESH")
    MEMORY_SIM_THRESHOLD: float = Field(0.65, env="MEMORY_SIM_THRESHOLD")


    # -----------------------------
    # Mongo / Database
    # -----------------------------
    MONGODB_PROVIDER: str = Field("local", env="MONGODB_PROVIDER")
    MONGODB_LOCAL_URI: str = Field("mongodb://localhost:27017/pathrag_ncec", env="MONGODB_LOCAL_URI")
    MONGODB_REMOTE_URI: Optional[str] = Field(None, env="MONGODB_REMOTE_URI")
    MONGODB_ENABLE_WEB_UI: bool = Field(True, env="MONGODB_ENABLE_WEB_UI")
    MONGODB_NAME: str = Field(..., env="MONGODB_NAME")

    # -----------------------------
    # Document storage / uploads
    # -----------------------------
    UPLOAD_ALLOWED_TYPES: List[str] = Field(
        default_factory=lambda: [".txt", ".pdf"],
        env="UPLOAD_ALLOWED_TYPES",
        description="Comma-separated list of allowed upload types (from .env)."
    )

    UPLOAD_MAX_FILE_SIZE_MB: int = Field(50, env="UPLOAD_MAX_FILE_SIZE_MB")
    UPLOAD_STORAGE_PATH: Path = Field(Path("./storage/documents"), env="UPLOAD_STORAGE_PATH")
    UPLOAD_TEMP_PATH: Path = Field(Path("./storage/temp"), env="UPLOAD_TEMP_PATH")
    DOC_LOCATION_STORE: Path = Field(Path("./assets/docs"), env="DOC_LOCATION_STORE")

    TEXT_CHUNK_SIZE: int = Field(512, ge=100, le=2000, env="TEXT_CHUNK_SIZE")
    TEXT_CHUNK_OVERLAP: int = Field(50, ge=0, le=100, env="TEXT_CHUNK_OVERLAP")
    TEXT_MAX_CHUNK_COUNT: int = Field(10000, env="TEXT_MAX_CHUNK_COUNT")

    # -----------------------------
    # Embedding / Model config
    # -----------------------------
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    EMBEDDING_DIMENSIONS: int = Field(384, env="EMBEDDING_DIMENSIONS")
    EMBEDDING_BATCH_SIZE: int = Field(32, env="EMBEDDING_BATCH_SIZE")
    EMBEDDING_DEVICE: str = Field("cpu", env="EMBEDDING_DEVICE")

    # Ollama
    OLLAMA_HOST: Optional[str] = Field(None, env="OLLAMA_HOST")
    OLLAMA_EMBEDDING_MODEL: Optional[str] = Field(None, env="OLLAMA_EMBEDDING_MODEL")
    OLLAMA_CHAT_MODEL: Optional[str] = Field(None, env="OLLAMA_CHAT_MODEL")

    BUILD_GRAPH_METHOD: str = Field(..., env="BUILD_GRAPH_METHOD")
    # -----------------------------
    # Generation / Inference
    # -----------------------------
    MAX_INPUT_TOKENS: int = Field(2048, env="MAX_INPUT_TOKENS")
    MAX_OUTPUT_TOKENS: int = Field(1024, env="MAX_OUTPUT_TOKENS")
    TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0, env="TEMPERATURE")
    TOP_K: int = Field(40, ge=1, le=1000, env="TOP_K")
    TOP_P: float = Field(0.95, ge=0.0, le=1.0, env="TOP_P")

    # -----------------------------
    # Graph / PathRAG
    # -----------------------------
    GRAPH_SIMILARITY_THRESHOLD: float = Field(0.75, ge=0.0, le=1.0, env="GRAPH_SIMILARITY_THRESHOLD")
    GRAPH_DECAY_RATE: float = Field(0.9, ge=0.1, le=0.9999, env="GRAPH_DECAY_RATE")
    GRAPH_PRUNING_THRESHOLD: float = Field(0.3, ge=0.0, le=1.0, env="GRAPH_PRUNING_THRESHOLD")
    PATH_MAX_DEPTH: int = Field(5, ge=1, le=50, env="PATH_MAX_DEPTH")

    # -----------------------------
    # Monitoring thresholds
    # -----------------------------
    MONITOR_CPU_THRESHOLD: float = Field(0.85, ge=0.0, le=1.0, env="MONITOR_CPU_THRESHOLD")
    MONITOR_MEMORY_THRESHOLD: float = Field(0.90, ge=0.0, le=1.0, env="MONITOR_MEMORY_THRESHOLD")
    MONITOR_DISK_THRESHOLD: float = Field(0.85, ge=0.0, le=1.0, env="MONITOR_DISK_THRESHOLD")
    MONITOR_GPU_THRESHOLD: float = Field(0.90, ge=0.0, le=1.0, env="MONITOR_GPU_THRESHOLD")
    MONITOR_HEALTH_CHECK_INTERVAL_SEC: int = Field(30, ge=1, env="MONITOR_HEALTH_CHECK_INTERVAL_SEC")

    # -----------------------------
    # KNN / retrieval
    # -----------------------------
    KNN_K_NEIGHBORS: int = Field(50, ge=1, le=1000, env="KNN_K_NEIGHBORS")
    KNN_SIMILARITY_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, env="KNN_SIMILARITY_THRESHOLD")
    KNN_INDEX_TYPE: str = Field("faiss", env="KNN_INDEX_TYPE")
    KNN_METRIC: str = Field("cosine", env="KNN_METRIC")

    # -----------------------------
    # Checkpoint / persistence
    # -----------------------------
    CHECKPOINT_ENABLED: bool = Field(True, env="CHECKPOINT_ENABLED")
    CHECKPOINT_DIRECTORY: Path = Field(Path("./storage/checkpoints"), env="CHECKPOINT_DIRECTORY")
    CHECKPOINT_GRAPH_FILE: str = Field("graph_state.pkl", env="CHECKPOINT_GRAPH_FILE")

    # -----------------------------
    # Logging
    # -----------------------------
    LOG_FORMAT: str = Field("json", env="LOG_FORMAT")
    LOG_FILE_PATH: Path = Field(Path("./logs/pathrag.log"), env="LOG_FILE_PATH")
    LOG_ENABLE_CONSOLE: bool = Field(True, env="LOG_ENABLE_CONSOLE")

    # -----------------------------
    # Misc / Feature flags
    # -----------------------------
    FEATURE_WEB_INTERFACE: bool = Field(True, env="FEATURE_WEB_INTERFACE")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # -----------------------------
    # Derived / convenience fields
    # -----------------------------
    FILE_TYPES: List[str] = Field(default_factory=lambda: [".txt", ".pdf"])

    # -----------------------------
    # Field validators
    # -----------------------------
    @field_validator("UPLOAD_ALLOWED_TYPES", mode="before")
    def _v_upload_allowed_types(cls, v):
        # Accept comma separated strings from .env
        parsed = _parse_list(v)
        return parsed or [".txt", ".pdf"]

    @field_validator("UPLOAD_STORAGE_PATH", "UPLOAD_TEMP_PATH", "DOC_LOCATION_STORE", "CHECKPOINT_DIRECTORY", "LOG_FILE_PATH", mode="before")
    def _v_path(cls, v):
        if v is None:
            return v
        return Path(v)

    @field_validator("MONGODB_ENABLE_WEB_UI", "CHECKPOINT_ENABLED", "LOG_ENABLE_CONSOLE", "FEATURE_WEB_INTERFACE", mode="before")
    def _v_bool(cls, v):
        # handle typical string boolean values from .env
        if isinstance(v, bool):
            return v
        parsed = _parse_bool(v)
        if parsed is None:
            # fallback: truthiness
            return bool(v)
        return parsed

    # -----------------------------
    # Pydantic Settings meta
    # -----------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings(env_file: Optional[str] = None) -> Settings:
    """
    Load and validate settings.

    If env_file is provided, it will override the default .env loaded by model_config.
    Ensures storage directories exist and logs any critical failures before exiting.
    """
    try:
        if env_file:
            # create a small Settings subclass to override model_config at runtime
            class _Settings(Settings):
                model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")
            settings = _Settings()
        else:
            settings = Settings()

        # Ensure directories exist (create parents)
        for p in (
            settings.UPLOAD_STORAGE_PATH,
            settings.UPLOAD_TEMP_PATH,
            settings.DOC_LOCATION_STORE,
            settings.CHECKPOINT_DIRECTORY,
            settings.LOG_FILE_PATH.parent,
        ):
            try:
                if p is None:
                    continue
                p.mkdir(parents=True, exist_ok=True)
                LOG.debug("Ensured directory exists: %s", p)
            except Exception as e:
                LOG.critical("Failed to create directory %s: %s", p, e)
                raise

        LOG.info("Loaded application settings successfully.")
        return settings

    except ValidationError as ve:
        LOG.critical("Configuration validation failed:\n%s", ve.model_dump_json(indent=2))
        # Provide exit for scripts; in other contexts you could raise.
        sys.exit(1)

    except Exception as e:
        LOG.critical("Unexpected error while loading settings: %s", e)
        sys.exit(1)


# Quick usage demo when run directly
if __name__ == "__main__":
    s = get_settings()
    LOG.info("UPLOAD_ALLOWED_TYPES: %s", s.UPLOAD_ALLOWED_TYPES)
    LOG.info("UPLOAD_STORAGE_PATH: %s", s.UPLOAD_STORAGE_PATH)
    LOG.info("MONGODB_LOCAL_URI: %s", s.MONGODB_LOCAL_URI)
