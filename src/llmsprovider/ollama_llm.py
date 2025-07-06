"""
OllamaLLM - Enhanced LLM Client for LightRAG System

Features:
- Robust Server Management
- Automatic Model Download
- Efficient Relation Extraction
- Concise Entity/Relation Description
- Accurate Keyword Extraction
- Context-Aware Answer Generation
- Complete Pipeline Implementation
"""

# pylint: disable=broad-exception-caught
# pylint: disable=redefined-outer-name
# pylint: disable=wrong-import-position
import os
import sys
import logging
import json
from typing import Any, Dict, List, Tuple, Optional

import requests

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Failed to set project root: %s", e)

from src.utils import setup_logging, OllamaManager


logger = setup_logging()

class OllamaModel:
    """
    Enhanced Ollama-based LLM interface with comprehensive error handling and optimization.
    """

    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        """
        Initializes the Ollama LLM client.

        Args:
            model_name (str): Name of the model to use.
            base_url (str): Base URL of the Ollama server.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.model = model_name
        self.host = "localhost"
        self.port = 11434

        try:
            manager = OllamaManager()
            manager.execute_workflow(model_name=model_name)
            logger.info("✅ Ollama model workflow executed successfully.")
        except Exception as e:
            logger.critical("❌ Model initialization failed: %s", e)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text using the specified Ollama model.

        Args:
            prompt (str): User prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature for generation.
            system_message (Optional[str]): Optional system-level message to steer the response.

        Returns:
            str: The generated response.
        """
        logger.debug("📝 Generating response (prompt length: %d)", len(prompt))

        full_prompt = f"{system_message.strip() if system_message else ''}\n\n{prompt}".strip()

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.RequestException as e:
            logger.error("❌ Request failed: %s", e)
            return "[ERROR] Request failed."
        except Exception as e:
            logger.error("❌ Text generation failed: %s", e)
            raise RuntimeError("Text generation error") from e

