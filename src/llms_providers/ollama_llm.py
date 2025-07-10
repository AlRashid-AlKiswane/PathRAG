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
import time
from typing import Optional

import requests

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Failed to set project root: %s", e)

from src.infra import setup_logging
from src.utils import OllamaManager


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
            
            # Wait for model to be ready
            self._wait_for_model_ready()
            
            logger.info("✅ Ollama model workflow executed successfully.")
        except Exception as e:
            logger.critical("❌ Model initialization failed: %s", e)

    def _wait_for_model_ready(self, max_wait_time: int = 30) -> None:
        """
        Wait for the model to be ready for requests.
        
        Args:
            max_wait_time (int): Maximum time to wait in seconds.
        """
        logger.info("⏳ Waiting for model to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Test with a simple request
                response = self._make_request("Hello", max_tokens=1, temperature=0.1)
                if response and response != "[ERROR] Request failed.":
                    logger.info("✅ Model is ready for requests")
                    return
            except Exception as e:
                logger.debug("Model not ready yet: %s", e)
            
            time.sleep(2)
        
        logger.warning("⚠️ Model may not be fully ready after %d seconds", max_wait_time)

    def _make_request(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Make a request to the Ollama API with error handling.
        
        Args:
            prompt (str): The prompt to send.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Temperature for generation.
            
        Returns:
            str: Generated response or error message.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=60  # Increased timeout
            )
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get("response", "").strip()
            if not generated_text:
                logger.warning("⚠️ Empty response from model")
                return "[ERROR] Empty response from model."
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out")
            return "[ERROR] Request timed out."
        except requests.exceptions.ConnectionError:
            logger.error("❌ Connection error - is Ollama server running?")
            return "[ERROR] Connection error."
        except requests.exceptions.HTTPError as e:
            logger.error("❌ HTTP error: %s", e)
            if e.response.status_code == 500:
                logger.error("Server error - model may not be loaded properly")
                return "[ERROR] Server error - model not ready."
            return f"[ERROR] HTTP error: {e.response.status_code}."
        except requests.RequestException as e:
            logger.error("❌ Request failed: %s", e)
            return "[ERROR] Request failed."
        except Exception as e:
            logger.error("❌ Unexpected error: %s", e)
            return "[ERROR] Unexpected error."

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        system_message: Optional[str] = None,
        retry_count: int = 3,
        max_input_tokens: int = 1024,
    ) -> str:
        """
        Generate text using the specified Ollama model with retry logic.

        Args:
            prompt (str): The user's prompt to the model.
            max_tokens (int): Maximum tokens to generate in the response.
            temperature (float): Sampling temperature for randomness.
            system_message (Optional[str]): Optional system-level instruction (prepended).
            retry_count (int): Number of retry attempts on failure.
            max_input_tokens (int): Maximum tokens allowed in input.

        Returns:
            str: The generated text, or an error string if generation fails.
        """
        if not prompt or not prompt.strip():
            logger.warning("⚠️ Empty prompt provided")
            return "[ERROR] Empty prompt."

        try:
            logger.debug("📝 Preparing prompt for generation (original length: %d)", len(prompt))

            # Build final prompt with optional system message
            full_prompt = f"{system_message.strip()}\n\n{prompt}".strip() if system_message else prompt.strip()

            # Truncate to max_input_tokens
            truncated_prompt = full_prompt[:max_input_tokens]

            logger.debug("📨 Final prompt length: %d (max allowed: %d)", len(truncated_prompt), max_input_tokens)

            for attempt in range(retry_count + 1):
                try:
                    logger.debug("🔁 Generation attempt %d", attempt + 1)
                    response = self._make_request(
                        prompt=truncated_prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature
                    )

                    if not response.startswith("[ERROR]"):
                        logger.info("✅ Text generation succeeded on attempt %d", attempt + 1)
                        return response

                    logger.warning("⚠️ Model returned error: %s", response)

                except Exception as e:
                    logger.exception("❌ Exception during generation attempt %d: %s", attempt + 1, str(e))

                # If not the last attempt, wait before retrying
                if attempt < retry_count:
                    logger.debug("⏳ Retrying after delay...")
                    time.sleep(2)

            logger.error("❌ Text generation failed after %d attempts", retry_count + 1)
            return "[ERROR] Text generation failed after all retries."

        except Exception as outer_error:
            logger.exception("💥 Unexpected error during prompt generation: %s", str(outer_error))
            return "[ERROR] Unexpected failure in text generation."

    def is_available(self) -> bool:
        """
        Check if the Ollama service is available.
        
        Returns:
            bool: True if available, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model information.
        """
        try:
            response = requests.get(f"{self.base_url}/api/show", json={"name": self.model_name}, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to get model info"}
        except Exception as e:
            logger.error("Failed to get model info: %s", e)
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Test the OllamaModel
    model = OllamaModel()
    
    # Test if service is available
    if model.is_available():
        print("✅ Ollama service is available")
        
        # Test generation
        response = model.generate("What is the capital of France?")
        print(f"Response: {response}")
    else:
        print("❌ Ollama service is not available")
