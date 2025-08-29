#!/usr/bin/env python3
"""
Auto-Save Integration for PathRAG

This module provides an AutoSave class that integrates seamlessly with a
PathRAG instance to support periodic checkpointing during long-running
graph building or retrieval tasks.

Features:
    - Auto-save during graph building every N samples
    - Time-based auto-save every N minutes
    - Resumable checkpoints with graph, FAISS index, and embeddings
    - Metadata persistence for progress tracking
    - Uses PathRAG's existing save_graph() and load_graph() methods

Intended Usage:
    >>> pathrag = PathRAG(...)
    >>> autosaver = AutoSave(pathrag, save_dir="./checkpoints")
    >>> for sample in data:
    >>>     pathrag.process(sample)
    >>>     autosaver.processed_samples += 1
    >>>     if autosaver.should_save():
    >>>         autosaver.save_checkpoint(reason="interval")
"""

import os
import sys
import json
import time
from pathlib import Path
import logging

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.schemas import CheckpointMetadata

logger = setup_logging(name="AUTO-SAVE-MANAGER")


class AutoSave:
    """
    Auto-save manager for PathRAG.

    This class manages automatic checkpointing of PathRAG graph state,
    FAISS index, and related metadata, enabling resumption after interruption.

    Attributes:
        pathrag (PathRAG): Instance of the PathRAG model being managed.
        save_dir (Path): Directory where checkpoints are stored.
        save_every_samples (int): Save every N processed samples.
        save_every_minutes (int): Save every N minutes.
        processed_samples (int): Current number of processed samples.
        total_samples (int): Total number of samples to process (if known).
        last_save_samples (int): Samples processed at last checkpoint.
        last_save_time (float): Epoch timestamp of last checkpoint.
        graph_file (Path): File path for the serialized graph checkpoint.
        metadata_file (Path): File path for JSON metadata.
        progress_file (Path): File path for progress tracking (optional).
    """

    def __init__(
        self,
        pathrag_instance,
        save_dir: str = "./pathrag_data/checkpoints",
        save_every_samples: int = 1000,
        save_every_minutes: int = 10,
    ):
        """
        Initialize auto-save manager.

        Args:
            pathrag_instance (PathRAG): Your PathRAG instance.
            save_dir (str): Directory to save checkpoints.
            save_every_samples (int): Save after this many processed samples.
            save_every_minutes (int): Save after this many minutes.
        """
        self.pathrag = pathrag_instance
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_samples = save_every_samples
        self.save_every_minutes = save_every_minutes

        # Progress tracking
        self.processed_samples: int = 0
        self.total_samples: int = 0
        self.last_save_samples: int = 0
        self.last_save_time: float = time.time()

        # File paths
        self.graph_file: Path = self.save_dir / "checkpoint_graph.pkl"
        self.metadata_file: Path = self.save_dir / "checkpoint_metadata.json"
        self.progress_file: Path = self.save_dir / "progress.json"

        logger.info("AutoSave initialized at directory: %s", str(save_dir))

    def should_save(self) -> bool:
        """
        Determine if a checkpoint should be saved.

        Returns:
            bool: True if checkpoint should be saved based on samples or time.
        """
        try:
            samples_diff = self.processed_samples - self.last_save_samples
            time_diff = (time.time() - self.last_save_time) / 60  # minutes

            return (
                samples_diff >= self.save_every_samples
                or time_diff >= self.save_every_minutes
            )
        except Exception as e:
            logger.error("Failed to evaluate save condition: %s", e)
            return False

    def save_checkpoint(self, reason: str = "auto") -> None:
        """
        Save a checkpoint including PathRAG graph, FAISS index, embeddings,
        and metadata.

        Args:
            reason (str): Reason for checkpointing (e.g., 'auto', 'manual').
        """
        try:
            # Save the graph via PathRAG
            self.pathrag.save_graph(
                file_path=self.graph_file, format="pickle", compress=True
            )

            # Save FAISS index if available
            faiss_file = self.save_dir / "faiss_index.bin"
            faiss_saved = False
            try:
                if hasattr(self.pathrag, "index") and self.pathrag.index is not None:
                    import faiss

                    faiss.write_index(self.pathrag.index, str(faiss_file))

                    # Save node IDs and embeddings if available
                    if hasattr(self.pathrag, "node_ids"):
                        import numpy as np

                        np.save(self.save_dir / "node_ids.npy", self.pathrag.node_ids)
                    if hasattr(self.pathrag, "embedding_matrix"):
                        np.save(
                            self.save_dir / "embeddings.npy",
                            self.pathrag.embedding_matrix,
                        )
                    faiss_saved = True
                    logger.info("FAISS index saved successfully.")
            except Exception as faiss_err:
                logger.warning("Failed to save FAISS index: %s", faiss_err)

            # Save checkpoint metadata
            checkpoint = CheckpointMetadata(
                processed_samples=self.processed_samples,
                total_samples=self.total_samples,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                graph_nodes=self.pathrag.g.number_of_nodes(),
                graph_edges=self.pathrag.g.number_of_edges(),
            )

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        **checkpoint.__dict__,
                        "faiss_saved": faiss_saved,
                        "reason": reason,
                    },
                    f,
                    indent=2,
                )

            # Update tracking
            self.last_save_samples = self.processed_samples
            self.last_save_time = time.time()

            logger.info(
                "Checkpoint saved. Samples=%d, Nodes=%d, Edges=%d",
                checkpoint.processed_samples,
                checkpoint.graph_nodes,
                checkpoint.graph_edges,
            )
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
