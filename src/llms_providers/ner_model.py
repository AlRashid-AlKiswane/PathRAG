"""
NER model wrapper for LightRAG system using a small Hugging Face model.
Supports multilingual named entity recognition (NER) with lightweight BERT variants.

Author: Alrashid AlKiswane
"""

import logging
import os
import sys
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers.pipelines import AggregationStrategy
import transformers
transformers.logging.set_verbosity_error()

# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# Pylint directive
# pylint: disable=wrong-import-position
from src.infra import setup_logging

logger = setup_logging()


class NERModel:
    """
    Named Entity Recognition (NER) model for LightRAG.
    
    Loads a small pre-trained model from Hugging Face Transformers,
    optimized for speed and memory efficiency.

    Attributes:
        device (str): 'cuda' if available, else 'cpu'.
        tokenizer: Pre-trained tokenizer for the NER model.
        model: Token classification model.
        pipeline: Hugging Face NER pipeline with aggregation.
    """

    def __init__(self, model_name: str = "dslim/bert-large-NER"):
        """
        Initialize the NERModel with tokenizer, model, and pipeline.

        Args:
            model_name (str): Model identifier from Hugging Face hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("NERModel device set to: %s", self.device)

        try:
            logger.info("Loading tokenizer and model from Hugging Face: %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy=AggregationStrategy.SIMPLE,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("NERModel successfully loaded.")
        except Exception as e:
            logger.error("Failed to load NER model: %s", e)
            raise RuntimeError(f"Error loading model '{model_name}': {str(e)}") from e

    def predict(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run NER prediction on the given text and return a debuplicated list of
        entity strings.

        Args:
            text (str): Input text to extract entities from.
        
        Returns:
            List[str]: Unique entity names extracted from the text.
        """
        logger.debug("Starting NER prediction...")
        if not text.strip():
            logger.warning("Input text is empty or whitespace.")
        
        try:
            raw_entities = self.pipeline(text.strip())
            logger.debug("NER raw output: %s", raw_entities)
        except Exception as e:
            logger.error("NER pipeline error: %s", e)
        
        seen = set()
        entity_texts = []

        for ent in raw_entities:
            try:
                entity_word = ent.get("word", "").strip()
                if not entity_word or entity_word.lower() in seen:
                    continue
                seen.add(entity_word.lower())
                entity_texts.append(entity_word)
                logger.debug("Entity extractted: %s", entity_word)
            except Exception as e:
                logger.warning("Skipping malformed entity: %s", e)
                continue
        return entity_texts

# Example usage
if __name__ == "__main__":
    logger.info("Running NERModel test with example input...")

    ner_model = NERModel()

    text = (
        "In 2024, OpenAI partnered with Microsoft and the World Health Organization to develop an AI-driven "
        "epidemiological modeling tool for pandemic forecasting. The tool was deployed in over 15 countries, "
        "including Germany, Brazil, and India. Funding was partially provided by the Gates Foundation."
    )

    result = ner_model.predict(text)
    
    print(result)
