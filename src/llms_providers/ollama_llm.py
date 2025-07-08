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
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_message: Optional[str] = None,
        retry_count: int = 3
    ) -> str:
        """
        Generate text using the specified Ollama model with retry logic.

        Args:
            prompt (str): User prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature for generation.
            system_message (Optional[str]): Optional system-level message.
            retry_count (int): Number of retries on failure.

        Returns:
            str: The generated response.
        """
        if not prompt or not prompt.strip():
            logger.warning("⚠️ Empty prompt provided")
            return "[ERROR] Empty prompt."

        logger.debug("📝 Generating response (prompt length: %d)", len(prompt))

        # Construct full prompt
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message.strip()}\n\n{prompt}".strip()

        # Try generating with retries
        for attempt in range(retry_count + 1):
            try:
                response = self._make_request(full_prompt, max_tokens, temperature)
                
                if not response.startswith("[ERROR]"):
                    logger.debug("✅ Successfully generated response")
                    return response
                    
                if attempt < retry_count:
                    logger.warning("⚠️ Attempt %d failed, retrying...", attempt + 1)
                    time.sleep(2)  # Wait before retry
                else:
                    logger.error("❌ All attempts failed")
                    return response
                    
            except Exception as e:
                logger.error("❌ Generation attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_count:
                    time.sleep(2)
                else:
                    return "[ERROR] Text generation failed after all retries."

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
