"""
Gemini LLM Client Module for LightRAG System.

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
import google.generativeai as genai
from pydantic import BaseModel

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


class EntityDescriptionRequest:
    text: str
    type: str


class RelationDescriptionRequest:
    source: str
    type: str
    target: str


class GeminiLLM:
    """
    Google Gemini LLM client implementing all LightRAG required operations.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            logger.error("Gemini API key not found in environment variables")
            raise ValueError("GEMINI_API_KEY environment variable required")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info("Initialized Gemini LLM with model: %s", self.model_name)

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

        try:
            response = self.client.generate_content(
                f"{system_prompt}\n\nText:\n{prompt}",
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512
                )
            )

            output = response.text.strip() if hasattr(response, "text") else ""
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

        except Exception as e:
            logger.error("Relation extraction failed: %s", e, exc_info=True)
            raise RuntimeError(f"Relation extraction failed: {e}") from e

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

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=128
                )
            )
            return response.text.strip() if hasattr(response, "text") else ""

        except Exception as e:
            logger.error("Entity description failed: %s", e, exc_info=True)
            raise RuntimeError(f"Entity description failed: {e}") from e

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

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=256
                )
            )
            return response.text.strip() if hasattr(response, "text") else ""

        except Exception as e:
            logger.error("Relation description failed: %s", e, exc_info=True)
            raise RuntimeError(f"Relation description failed: {e}") from e

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

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=256
                )
            )

            local = []
            global_ = []
            output = response.text.strip() if hasattr(response, "text") else ""

            for line in output.splitlines():
                if line.startswith("Local:"):
                    local = [kw.strip() for kw in line.split(":", 1)[1].split(",") if kw.strip()]
                elif line.startswith("Global:"):
                    global_ = [kw.strip() for kw in line.split(":", 1)[1].split(",") if kw.strip()]

            return local, global_

        except Exception as e:
            logger.error("Keyword extraction failed: %s", e, exc_info=True)
            raise RuntimeError(f"Keyword extraction failed: {e}") from e

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

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=512
                )
            )
            return response.text.strip() if hasattr(response, "text") else ""

        except Exception as e:
            logger.error("Answer generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Answer generation failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities."""
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini",
            "capabilities": {
                "relation_extraction": True,
                "description_generation": True,
                "keyword_extraction": True,
                "answer_generation": True
            }
        }
