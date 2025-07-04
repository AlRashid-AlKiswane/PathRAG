"""
Language Model Interface Module.

This module defines an abstract base class (ABC) for Large Language Model (LLM) services.
It provides a standardized interface for different LLM implementations, ensuring consistent
behavior across various model providers and implementations.

The BaseLLM class serves as a contract that all concrete LLM implementations must follow,
with required methods for generating responses and providing model metadata.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


# pylint: disable=unnecessary-pass
class BaseLLM(ABC):
    """
    Abstract base class for all LLM service implementations.

    This class defines the core interface that all language model implementations must follow.
    Subclasses should provide concrete implementations for generating responses and reporting
    model capabilities.

    Attributes:
        None (abstract base class)
    """

    @abstractmethod
    def _generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
        """
        
        """
        pass

    def extract_relations(self, prompt: str) -> List[Dict[str, str]]:
        """
        Extract relationships between entities from the text.

        Args:
            prompt: Text containing entities and potential relations.

        Returns:
            List of dictionaries with keys: 'source', 'type', 'target'.
        """
        pass

    def describe_entity(self, entity_text: str, entity_type: str) -> str:
        """
        Generate a description of an entity.

        Args:
            entity_text: Name/text of the entity.
            entity_type: Type/category of the entity.

        Returns:
            Description string.
        """
        pass

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
        pass

    def extract_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract local and global keywords from the text.

        Args:
            text: Input text.

        Returns:
            Tuple containing (local_keywords, global_keywords).
        """

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
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities."""
        pass
