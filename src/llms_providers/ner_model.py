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

    def __init__(self, model_name: str = "dslim/distilbert-NER"):
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
        Run NER prediction on the given text and return deduplicated entities in LightRAG format.

        Args:
            text (str): Input text to extract entities from.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Deduplicated entities in LightRAG-compatible format.
        """
        logger.debug("Starting NER prediction...")
        if not text.strip():
            logger.warning("Input text is empty or whitespace.")
            return {"entities": []}

        try:
            raw_entities = self.pipeline(text)
            logger.debug("NER raw output: %s", raw_entities)
        except Exception as e:
            logger.error("NER pipeline error: %s", e)
            return {"entities": []}

        seen = set()
        entities = []

        for ent in raw_entities:
            try:
                key = (ent["word"], ent["entity_group"], ent["start"], ent["end"])
                if key in seen:
                    logger.debug("Duplicate entity skipped: %s", key)
                    continue
                seen.add(key)
                entity = {
                    "text": ent["word"],
                    "type": ent["entity_group"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "score": ent["score"],
                    "source_text": text
                }
                entities.append(entity)
                logger.debug("Entity added: %s", entity)
            except KeyError as e:
                logger.warning("Skipping malformed entity: %s", e)
                continue

        return {"entities": entities}



# Example usage
if __name__ == "__main__":
    logger.info("Running NERModel test with example input...")

    ner_model = NERModel()

    sample_text = """
    Can you explain how Google DeepMind’s AlphaFold uses transformer-based
    architectures similar to BERT to predict protein structures, and how this
    compares to traditional bioinformatics approaches developed at institutions 
    like Stanford University and MIT? Also, what role did the CASP14 competition and 
    the EMBL-EBI play in validating the effectiveness of AlphaFold’s predictions, and
    how does this model relate to other recent innovations in generative AI such as
    OpenAI’s GPT-4 or Meta’s LLaMA models?
    """

    results = ner_model.predict(sample_text)

    for entity in results["entities"]:
        print(f"{entity['text']} - {entity['type']} (Confidence: {entity['score']:.2f})")
        print(f"Position: {entity['start']}-{entity['end']}")
        print(f"Source text: '{entity['source_text'][entity['start']:entity['end']]}'\n")
