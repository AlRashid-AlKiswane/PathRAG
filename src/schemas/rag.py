"""
PathRAG Configuration Module
----------------------------

This module defines the configuration class for the Path-aware Retrieval-Augmented 
Generation (PathRAG) system. The configuration controls graph-building, 
retrieval, caching, and resource usage settings. 

The `PathRAGConfig` class ensures all parameters are validated at initialization 
and provides a central location to manage system-wide constants.
"""

from pydantic import BaseModel, Field, field_validator


class PathRAGConfig(BaseModel):
    """
    Configuration settings for the PathRAG system.

    Attributes
    ----------
    decay_rate : float
        Decay rate for scoring paths, must be in (0, 1].
    prune_thresh : float
        Threshold for pruning paths; must be non-negative.
    sim_threshold : float
        Similarity threshold for connecting nodes; must be in [0, 1].
    max_workers : int
        Maximum number of parallel workers; must be positive.
    batch_size : int
        Number of items processed in a batch; must be positive.
    cache_size : int
        Maximum size of cache (in items).
    memory_limit_gb : float
        Maximum memory usage allowed for graph storage (in GB).
    enable_caching : bool
        Whether caching is enabled for retrieval results.
    max_graph_size : int
        Maximum number of nodes allowed in the graph.
    checkpoint_interval : int
        Interval (in steps) for saving checkpoints.
    """

    decay_rate: float = Field(..., description="Decay rate for path scoring, must be in (0, 1].")
    prune_thresh: float = Field(..., description="Threshold for pruning paths, non-negative.")
    sim_threshold: float = Field(..., description="Similarity threshold in [0,1].")
    max_workers: int = Field(..., description="Maximum number of parallel workers.")
    batch_size: int = Field(..., description="Batch size for processing.")
    cache_size: int = Field(..., description="Maximum number of cached items.")
    memory_limit_gb: float = Field(..., description="Memory limit in GB.")
    enable_caching: bool = Field(..., description="Enable/disable caching.")
    max_graph_size: int = Field(..., description="Maximum number of graph nodes.")
    checkpoint_interval: int = Field(..., description="Steps between checkpoints.")

    # ----------------- Validators -----------------
    @field_validator("decay_rate")
    @classmethod
    def validate_decay_rate(cls, v: float) -> float:
        """Ensure decay_rate is in (0, 1]."""
        if not (0 < v <= 1):
            raise ValueError("decay_rate must be in (0, 1]")
        return v

    @field_validator("prune_thresh")
    @classmethod
    def validate_prune_thresh(cls, v: float) -> float:
        """Ensure prune_thresh is non-negative."""
        if v < 0:
            raise ValueError("prune_thresh must be non-negative")
        return v

    @field_validator("sim_threshold")
    @classmethod
    def validate_sim_threshold(cls, v: float) -> float:
        """Ensure sim_threshold is in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError("sim_threshold must be in [0, 1]")
        return v

    @field_validator("max_workers", "batch_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure integer fields are strictly positive."""
        if v < 1:
            raise ValueError("Value must be positive")
        return v
