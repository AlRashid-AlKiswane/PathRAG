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
from datetime import datetime
import os
import sys
import logging
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
            logger.info("✅ Ollama model workflow executed successfully.")
        except Exception as e:
            logger.critical("❌ Model initialization failed: %s", e)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
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

    def extract_relation_between_entities(
        self,
        entity_1: str,
        entity_2: str,
        context: str,
        system_message: Optional[str] = "You are an expert in information extraction. Determine the semantic relation between two entities in the given context."
    ) -> str:
        """
        Extract the semantic relation between two entities using the LLM.

        Args:
            entity_1 (str): The first entity.
            entity_2 (str): The second entity.
            context (str): Context text that contains or surrounds both entities.
            system_message (Optional[str]): Instruction for the LLM to guide its response.

        Returns:
            str: The relation between the two entities.
        """
        prompt = (
            "You are an expert in extracting relationships between entities from a given context.\n"
            "For each example below, identify the relation between the two specified entities based on the context.\n\n"

            "Example 1:\n"
            "Context: 'Elon Musk founded SpaceX.'\n"
            "Entity 1: 'Elon Musk'\n"
            "Entity 2: 'SpaceX'\n"
            "Relation: 'founded'\n\n"

            "Example 2:\n"
            "Context: 'The Eiffel Tower is located in Paris.'\n"
            "Entity 1: 'Eiffel Tower'\n"
            "Entity 2: 'Paris'\n"
            "Relation: 'located in'\n\n"

            "Now, analyze the following example carefully and provide the relation between the two entities.\n"
            "After that, on a scale from 1 to 5, indicate your confidence level in the relation you provided, where 1 means low confidence and 5 means very high confidence.\n\n"

            f"Context:\n{context.strip()}\n\n"
            f"Entity 1: '{entity_1}'\n"
            f"Entity 2: '{entity_2}'\n\n"
            "Relation:"
            "\n\nConfidence (1-5):"
        )

        logger.info("🔍 Extracting relation between '%s' and '%s'", entity_1, entity_2)
        try:
            relation = self.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.2,
                system_message=system_message
            )
            logger.info("✅ Extracted relation: %s", relation)
            return {
                "entity_1": entity_1,
                "entity_2": entity_2,
                "relation": relation,
                "source": "ollama-gemma3",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("❌ Failed to extract relation: %s", e)
            return "[ERROR] Failed to extract relation."

if __name__ == "__main__":
    model = OllamaModel()

    test_cases = [
        # Leadership
        {
            "entity1": "Angela Merkel",
            "entity2": "Germany",
            "context": "Angela Merkel was the Chancellor of Germany from 2005 to 2021."
        },
        # Authorship
        {
            "entity1": "J.K. Rowling",
            "entity2": "Harry Potter",
            "context": "J.K. Rowling is the author of the globally successful Harry Potter book series."
        },
        # Birthplace
        {
            "entity1": "Albert Einstein",
            "entity2": "Germany",
            "context": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire."
        },
        # Company Founding
        {
            "entity1": "Steve Jobs",
            "entity2": "Apple",
            "context": "Steve Jobs co-founded Apple in 1976 along with Steve Wozniak and Ronald Wayne."
        },
        # Membership
        {
            "entity1": "Serena Williams",
            "entity2": "WTA",
            "context": "Serena Williams was a dominant player in the Women's Tennis Association (WTA) for decades."
        },
        # Headquartered Location
        {
            "entity1": "UNICEF",
            "entity2": "New York",
            "context": "UNICEF is headquartered in New York and operates worldwide to support children's rights."
        },
        # Product Development
        {
            "entity1": "OpenAI",
            "entity2": "ChatGPT",
            "context": "ChatGPT is a language model developed by OpenAI to assist with natural language processing tasks."
        },
        # Awards
        {
            "entity1": "Malala Yousafzai",
            "entity2": "Nobel Peace Prize",
            "context": "Malala Yousafzai received the Nobel Peace Prize for her advocacy of girls' education."
        },
        # Scientific Contribution
        {
            "entity1": "Isaac Newton",
            "entity2": "Gravity",
            "context": "Isaac Newton developed the theory of gravity after observing a falling apple."
        },
        # City-Country
        {
            "entity1": "Tokyo",
            "entity2": "Japan",
            "context": "Tokyo is the capital city of Japan and one of the most populous urban areas in the world."
        },
        # CEO of Tech Company
        {
            "entity1": "Satya Nadella",
            "entity2": "Microsoft",
            "context": "Satya Nadella became CEO of Microsoft in 2014, leading major innovations and acquisitions."
        },
        # Musical Group
        {
            "entity1": "Freddie Mercury",
            "entity2": "Queen",
            "context": "Freddie Mercury was the lead vocalist of the rock band Queen, known for his powerful performances."
        },
        # Film Role
        {
            "entity1": "Leonardo DiCaprio",
            "entity2": "Titanic",
            "context": "Leonardo DiCaprio played the role of Jack Dawson in the movie Titanic."
        },
        # Invention
        {
            "entity1": "Tim Berners-Lee",
            "entity2": "World Wide Web",
            "context": "Tim Berners-Lee is credited with inventing the World Wide Web in 1989."
        },
        # Headquarters and Country
        {
            "entity1": "Samsung",
            "entity2": "South Korea",
            "context": "Samsung, a multinational conglomerate, is headquartered in South Korea."
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Entities: {case['entity1']} <--> {case['entity2']}")
        print(f"Context: {case['context']}")
        relation = model.extract_relation_between_entities(
            case["entity1"], case["entity2"], case["context"]
        )
        print("🧩 Extracted Relation:", relation)

