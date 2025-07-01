"""
Language Model Interface Module.

This module defines an abstract base class (ABC) for Large Language Model (LLM) services.
It provides a standardized interface for different LLM implementations, ensuring consistent
behavior across various model providers and implementations.

The BaseLLM class serves as a contract that all concrete LLM implementations must follow,
with required methods for generating responses and providing model metadata.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


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
    def generate_response(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt: The user input prompt to generate a response for.
            **kwargs: Additional model-specific parameters that may include:
                     - temperature: Controls randomness
                     - max_tokens: Maximum length of response
                     - top_p: Nucleus sampling parameter
                     - etc.

        Returns:
            The generated response as a string.

        Raises:
            NotImplementedError: If not implemented by subclass
            RuntimeError: For model-specific errors during generation
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the model implementation.

        The returned dictionary should typically include:
        - model_name: Identifier for the model
        - provider: Service or framework providing the model
        - capabilities: Dictionary of supported features
        - parameters: Available configuration options

        Returns:
            Dictionary containing model metadata and capabilities.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
