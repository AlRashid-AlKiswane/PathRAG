"""
Sentence Transformer Embedding Module.

This module provides a wrapper class for generating texts embeddings using
SentenceTransformer models. It handles model initialization, texts embedding,
and includes error handling and logging capabilities.
"""

import logging
import os
import sys
from typing import List, Union, Optional
# pylint: disable=import-error
from sentence_transformers import SentenceTransformer  # type: ignore

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings

# Initialize application settings and logger
logger = setup_logging()
app_settings: Settings = get_settings()


class HuggingFaceModel:
    """
    Handles SentenceTransformer-based embeddings.

    This class provides functionality to:
    - Load a pre-trained SentenceTransformer model
    - Generate embeddings for texts inputs
    - Handle errors during embedding generation
    - Log embedding operations

    Args:
        model_name: Name of the SentenceTransformer model to load.
                   If None, uses the model from app settings.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding model."""
        self.model_name = model_name or app_settings.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model '%s' initialized successfully.", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize SentenceTransformer model '%s': %s", self.model_name, str(e))
            raise

    def embed_texts(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Generate embeddings for a given string or list of strings.

        Args:
            texts: Single string or list of strings to embed.
            convert_to_tensor: Whether to return embeddings as tensors.
            normalize_embeddings: Whether to normalize the embeddings.

        Returns:
            Embeddings as a list or tensor, or None on error.

        Raises:
            ValueError: If input texts is empty or invalid.
            Exception: For other embedding generation errors.
        """
        if not texts:
            error_msg = "Input texts for embedding are empty or invalid."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            embedding = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings
            )

            preview_texts = texts if isinstance(texts, str) else texts[0]
            logger.info("Embedding generated successfully for: %s", preview_texts)
            return embedding.tolist()
        # pylint: disable=broad-exception-caught
        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            return None

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information.
        """
        logger.info("Retrieved SentenceTransformer model information.")
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }


if __name__ == "__main__":
    # Example usage
    embedd = HuggingFaceModel()
    sample_text = "This"
    embedding = embedd.embed_texts(sample_text)
    print("Embedding: %s" % embedding)
