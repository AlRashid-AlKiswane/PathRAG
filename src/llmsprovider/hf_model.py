"""
HuggingFace LLM Client Module for LightRAG System.

Implements all required LLM operations for:
- Relation extraction
- Entity/relation description
- Keyword extraction
- Answer generation
"""

import logging
import os
import sys
from typing import Dict, Any, List, Tuple
import gc
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.utils import setup_logging
from src.llmsprovider.abcllm import BaseLLM

logger = setup_logging()

class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace LLM client implementing all LightRAG required operations.
    Uses Gemma 3.21B model.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-Prompt-Guard-2-22M") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Initialize model and tokenizer
        self._initialize_model()
        logger.info("Initialized HuggingFace LLM with model: %s on device: %s",
                   self.model_name, self.device)

    def _initialize_model(self):
        """Initialize the model, tokenizer, and pipeline."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        except Exception as e:
            logger.error("Failed to initialize HuggingFace model: %s", e, exc_info=True)
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
        """Generate text using the HuggingFace pipeline."""
        try:
            # Format prompt for Gemma
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            # Extract generated text
            generated_text = response[0]['generated_text'].strip()

            # Clean up the response (remove any remaining special tokens)
            if "<end_of_turn>" in generated_text:
                generated_text = generated_text.split("<end_of_turn>")[0].strip()

            return generated_text

        except Exception as e:
            logger.error("HuggingFace generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"HuggingFace generation failed: {e}") from e

    def extract_relations(self, prompt: str) -> List[Dict[str, str]]:
        """
        Extract relationships between entities from the text.

        Args:
            prompt: Text containing entities and potential relations.

        Returns:
            List of dictionaries with keys: 'source', 'type', 'target'.
        """
        system_prompt = """Extract relationships between entities in the following format:
        - [Entity1] [Relationship] [Entity2]
        Return only the relationships in this exact format, one per line."""

        full_prompt = f"{system_prompt}\n\nText:\n{prompt}"
        output = self._generate(full_prompt, max_tokens=512, temperature=0.3)

        relations = []
        for line in output.splitlines():
            if '-' in line:
                parts = line.strip('- ').split()
                if len(parts) >= 3:
                    relations.append({
                        'source': parts[0],
                        'type': ' '.join(parts[1:-1]),
                        'target': parts[-1]
                    })

        return relations

    def describe_entity(self, entity_text: str, entity_type: str) -> str:
        """
        Generate a description of an entity.

        Args:
            entity_text: Name/text of the entity.
            entity_type: Type/category of the entity.

        Returns:
            Description string.
        """
        prompt = f"""Provide a concise 1-2 sentence description of this {entity_type}:
        {entity_text}

        Description:"""
        return self._generate(prompt, max_tokens=128, temperature=0.5)

    def describe_relation(self, source: str, relation_type: str, target: str) -> str:
        """
        Generate a description of a relationship between two entities.

        Args:
            source: Source entity.
            relation_type: Type of relation.
            target: Target entity.

        Returns:
            Description string.
        """
        prompt = f"""Explain the relationship '{relation_type}' between {source} and {target} 
        in 1-2 sentences, focusing on how they are connected.

        Explanation:"""
        return self._generate(prompt, max_tokens=256, temperature=0.4)

    def extract_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract local and global keywords from the text.

        Args:
            text: Input text.

        Returns:
            Tuple containing (local_keywords, global_keywords).
        """
        prompt = f"""Analyze this text and extract two types of keywords:
        1. Local/specific terms (names, exact concepts)
        2. Global/general terms (categories, abstract concepts)

        Text: {text}

        Format response exactly as:
        Local: comma, separated, terms
        Global: comma, separated, terms"""

        output = self._generate(prompt, max_tokens=256, temperature=0.2)

        local = []
        global_ = []

        for line in output.splitlines():
            if line.startswith("Local:"):
                local = [kw.strip() for kw in line.split(":", 1)[1].split(",") if kw.strip()]
            elif line.startswith("Global:"):
                global_ = [kw.strip() for kw in line.split(":", 1)[1].split(",") if kw.strip()]

        return local, global_

    def generate_answer(self, query: str,
                        entities: List[Dict[str, Any]],
                        relations: List[Dict[str, Any]]) -> str:
        """
        Generate a context-aware answer using entities and relations.

        Args:
            query: Original question.
            entities: List of entity dictionaries.
            relations: List of relation dictionaries.

        Returns:
            Answer string.
        """
        context = "Relevant Knowledge:\n"

        if entities:
            context += "Entities:\n" + "\n".join(
                f"- {e['text']} ({e['type']}): {e.get('description', '')}" for e in entities
            ) + "\n"

        if relations:
            context += "\nRelationships:\n" + "\n".join(
                f"- {r['source']} --{r['type']}--> {r['target']}: {r.get('description', '')}"
                for r in relations
            ) + "\n"

        prompt = f"""Using the provided context, answer the question concisely.
        Question: {query}
        {context}
        Answer:"""

        return self._generate(prompt, max_tokens=512, temperature=0.5)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities."""
        return {
            "model_name": self.model_name,
            "provider": "HuggingFace",
            "device": self.device,
            "capabilities": {
                "relation_extraction": True,
                "description_generation": True,
                "keyword_extraction": True,
                "answer_generation": True
            }
        }

    def __clear__(self):
        """Clear the model and free up memory."""
        try:
            logger.info("Clearing HuggingFace model from memory")

            # Clear pipeline
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None

            # Clear model
            if self.model is not None:
                del self.model
                self.model = None

            # Clear tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("HuggingFace model cleared successfully")

        except Exception as e:
            logger.error("Error clearing HuggingFace model: %s", e, exc_info=True)

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            logger.info("HuggingFace LLM destructor called")
            self.__clear__()
        except Exception as e:
            logger.error("Error in HuggingFace LLM destructor: %s", e, exc_info=True)

if __name__ == "__main__":
    llm = HuggingFaceLLM(model_name="meta-llama/Llama-Prompt-Guard-2-22M")

    # Sample input text (e.g., from a research paper abstract)
    sample_text = """
    Vitamin D plays a crucial role in bone health by regulating calcium absorption.
    Recent studies show that Vitamin D deficiency is linked to osteoporosis.
    Calcium is essential for bone mineralization and strength.
    """

    # Step 1: Extract relations from the text
    relations = llm.extract_relations(sample_text)
    print("Extracted Relations:")
    for rel in relations:
        print(rel)

    # Step 2: Describe a specific entity (e.g., Vitamin D)
    entity_description = llm.describe_entity("Vitamin D", "nutrient")
    print("\nEntity Description:\n", entity_description)

    # Step 3: Describe a relationship between two entities
    relation_description = llm.describe_relation("Vitamin D", "linked to", "osteoporosis")
    print("\nRelation Description:\n", relation_description)

    # Step 4: Extract local and global keywords
    local_keywords, global_keywords = llm.extract_keywords(sample_text)
    print("\nLocal Keywords:", local_keywords)
    print("Global Keywords:", global_keywords)

    # Step 5: Generate an answer based on entities and relations
    entities = [
        {"text": "Vitamin D", "type": "nutrient", "description": entity_description},
        {"text": "Calcium", "type": "mineral", "description": "Essential for bone strength"},
        {"text": "Osteoporosis", "type": "disease", "description": "A condition of fragile bones"}
    ]

    # Add descriptions to relations
    for rel in relations:
        rel["description"] = llm.describe_relation(rel["source"], rel["type"], rel["target"])

    query = "How does Vitamin D impact bone health?"

    answer = llm.generate_answer(query, entities, relations)
    print("\nGenerated Answer:\n", answer)

    # Optional: Clean up resources
    llm.__clear__()
