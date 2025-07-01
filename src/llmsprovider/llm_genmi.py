"""
Gemini LLM Client Module.

This module implements the BaseLLM interface for interacting with
Google's Gemini Large Language Models via the Generative AI API.
"""

import logging
import os
import sys
from typing import Dict, Any
import google.generativeai as genai

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.utils import setup_logging
from src.llmsprovider.abcllm import BaseLLM

# Initialize logger and settings
logger = setup_logging()


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM client implementing the BaseLLM interface.

    Handles text generation and model metadata retrieval using the Google Gemini API.
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini LLM with an optional custom model name.

        Args:
            model_name: The specific Gemini model name to use (e.g., "gemini-pro").
                        Defaults to the value in app_settings.
        """
        self.model_name = "gemini-1.5-flash"
        self.api_key = "AIzaSyAtz4Ae5hwBKOzihHTdhSWnL6CJ3HWqD78"

        if not self.api_key:
            logger.error("Gemini API key is not set correctly.")
            raise ValueError("Valid Gemini API key is required.")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info("Gemini LLM initialized with model: %s", self.model_name)

    def generate_response(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """
        Generate a response using the Gemini model.

        Args:
            prompt: Input prompt text.
            chat_history: Optional list of previous messages.
            **kwargs: Optional parameters like temperature, max_tokens, top_p, top_k.

        Returns:
            Generated response text.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: On API failure.
        """
        if not prompt:
            logger.error("Prompt is empty.")
            raise ValueError("Prompt must not be empty.")

        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 256),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 32)
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        try:
            # pylint: disable=redefined-outer-name
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config)
            )

            if response.candidates and response.candidates[0].content.parts:
                reply = response.candidates[0].content.parts[0].text.strip()
                logger.info("Gemini response generated.")
                return reply

            logger.warning("Empty response from Gemini.")
            return ""

        except Exception as e:
            logger.error("Gemini API call failed: %s", e, exc_info=True)
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return Gemini model metadata.

        Returns:
            Dictionary with model information and capabilities.
        """
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini",
            "capabilities": {
                "chat": True,
                "streaming": True,
                "temperature_control": True,
                "max_tokens": True,
                "top_p_control": True,
                "top_k_control": True
            },
            "parameters": {
                "temperature": "float [0.0 - 1.0]",
                "max_tokens": "int (max_output_tokens)",
                "top_p": "float [0.0 - 1.0]",
                "top_k": "int"
            }
        }

if __name__ == "__main__":
    model = GeminiLLM()
    response = model.generate_response(prompt="What is the AI?")
    print(response)
