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
import gc
import json
from typing import Any, Dict, List, Tuple, Optional

import requests
import torch
import ollama

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Failed to set project root: %s", e)

from src.utils import setup_logging, OllamaManager


logger = setup_logging()

class OllamaLLM:
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

    def _generate(
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

    def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships from text with enhanced parsing and fallback mechanisms.
        
        Args:
            text: Input text to analyze for relationships
            
        Returns:
            List of dictionaries with 'source', 'type', and 'target' keys.
            Returns empty list on failure.
        """
        logger.info("🔍 Extracting relations from text")

        # System message instructing the LLM on output expectations
        system_message = (
            "You are an expert at extracting relationships from text. "
            "Extract only clear, factual relationships. Respond ONLY with valid JSON.\n"
            "Example format:\n"
            '[{"source": "entity1", "type": "relationship", "target": "entity2"}]\n'
            'Do not include any explanatory text outside the JSON structure.'
        )

        # Prompt passed to the LLM
        prompt = f"""Extract all meaningful relationships from the following text.
        
        RULES:
        1. Only include relationships that are explicitly stated or clearly implied
        2. Format as a JSON array of objects
        3. Each object must have:
        - "source": the source entity (string)
        - "type": the relationship type (verb or descriptor)
        - "target": the target entity (string)
        4. Clean all strings (remove extra whitespace, quotes)
        5. Return ONLY the JSON array, no other text

        TEXT TO ANALYZE:
        {text}

        YOUR JSON OUTPUT:"""

        try:
            # Generate the model output
            output = self._generate(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                system_message=system_message
            )

            # Clean up code block markers if present
            cleaned_output = output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:-3].strip()
            elif cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[3:-3].strip()

            # Try parsing the JSON output
            try:
                relations_data = json.loads(cleaned_output)
                if not isinstance(relations_data, list):
                    raise ValueError("Top-level structure is not a list")

                relations = []
                for rel in relations_data:
                    if not all(key in rel for key in ['source', 'type', 'target']):
                        logger.warning("⚠️ Missing required keys in relation: %s", rel)
                        continue

                    relations.append({
                        'source': str(rel['source']).strip().strip('"\''),
                        'type': str(rel['type']).strip().strip('"\''),
                        'target': str(rel['target']).strip().strip('"\'')
                    })

                logger.info("✅ Extracted %d relations from JSON", len(relations))
                return relations

            except (json.JSONDecodeError, ValueError) as json_err:
                logger.error("❌ Failed to parse JSON: %s", json_err)
                return []

        except Exception as e:
            logger.error("❌ Relation extraction failed: %s", e)
            return []

    def describe_entity(self, entity_text: str, entity_type: str = "entity") -> str:
        """
        Generate a concise description for an entity.
        
        Args:
            entity_text: The entity to describe
            entity_type: Type of entity (optional)
            
        Returns:
            A concise 1-2 sentence description
        """
        logger.info("📘 Describing entity: %s", entity_text)

        system_message = (
            "You are an expert at providing concise, accurate descriptions. "
            "Provide factual information in 1-2 sentences."
        )

        prompt = f"""Provide a precise, factual description of the following {entity_type}:

                {entity_text}
                
                Description (1-2 sentences):"""

        try:
            description = self._generate(
                prompt,
                max_tokens=128,
                temperature=0.3,
                system_message=system_message
            )
            return description.strip()
        except Exception as e:
            logger.error("❌ Entity description failed: %s", e)
            return f"Unable to describe {entity_text}"

    def describe_relation(self, source: str, relation_type: str, target: str) -> str:
        """
        Generate a description of a relationship.
        
        Args:
            source: Source entity
            relation_type: Type of relationship
            target: Target entity
            
        Returns:
            A concise explanation of the relationship
        """
        logger.info("🔗 Describing relation: %s --%s--> %s", source, relation_type, target)

        system_message = (
            "You are an expert at explaining relationships. "
            "Provide clear, factual explanations in 1-2 sentences."
        )

        prompt = f"""Explain the relationship between these entities:

        Source: {source}
        Relationship: {relation_type}
        Target: {target}

        Explanation (1-2 sentences):"""

        try:
            description = self._generate(
                prompt,
                max_tokens=256,
                temperature=0.3,
                system_message=system_message
            )
            return description.strip()
        except Exception as e:
            logger.error("❌ Relation description failed: %s", e)
            return f"Unable to describe the {relation_type} relationship"

    def extract_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract local and global keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (local_keywords, global_keywords)
        """
        logger.info("🧠 Extracting keywords")

        system_message = (
            "You are an expert at keyword extraction. "
            "Identify specific terms (local) and general concepts (global)."
        )

        prompt = f"""Extract keywords from the following text in two categories:

        1. LOCAL KEYWORDS: Specific names, terms, entities, proper nouns
        2. GLOBAL KEYWORDS: General concepts, abstract terms, categories, themes

        Text: {text}

        Format your response as:
        LOCAL: keyword1, keyword2, keyword3
        GLOBAL: concept1, concept2, concept3"""

        try:
            output = self._generate(
                prompt,
                max_tokens=256,
                temperature=0.2,
                system_message=system_message
            )

            local_keywords = []
            global_keywords = []

            for line in output.split('\n'):
                line = line.strip()
                if line.lower().startswith('local:'):
                    keywords = line.split(':', 1)[1].strip()
                    local_keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                elif line.lower().startswith('global:'):
                    keywords = line.split(':', 1)[1].strip()
                    global_keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]

            logger.info("✅ Extracted %d local, %d global keywords",
                       len(local_keywords), len(global_keywords))
            return local_keywords, global_keywords

        except Exception as e:
            logger.error("❌ Keyword extraction failed: %s", e)
            return [], []

    def generate_answer(self,
                       query: str,
                       entities: List[Dict[str, Any]] = None,
                       relations: List[Dict[str, Any]] = None,
                       max_length: int = 512) -> str:
        """
        Generate an answer to a query using provided context.
        
        Args:
            query: The question to answer
            entities: List of relevant entities with descriptions
            relations: List of relevant relations with descriptions
            max_length: Maximum length of response
            
        Returns:
            A comprehensive answer based on the provided context
        """
        logger.info("💬 Generating answer for query: %s", query[:100])

        system_message = (
            "You are a knowledgeable assistant that provides accurate, "
            "evidence-based answers using only the provided context. "
            "Be concise but comprehensive."
        )

        context_parts = []

        if entities:
            context_parts.append("RELEVANT ENTITIES:")
            for entity in entities:
                entity_info = f"- {entity.get('text', 'Unknown')}"
                if entity.get('type'):
                    entity_info += f" ({entity['type']})"
                if entity.get('description'):
                    entity_info += f": {entity['description']}"
                context_parts.append(entity_info)

        if relations:
            context_parts.append("\nRELEVANT RELATIONSHIPS:")
            for relation in relations:
                rel_info = f"""- {
                    relation.get('source', 'Unknown')
                    } --{
                        relation.get('type', 'related to')
                        }--> {
                            relation.get('target', 'Unknown')
                            }"""
                if relation.get('description'):
                    rel_info += f": {relation['description']}"
                context_parts.append(rel_info)

        context = '\n'.join(context_parts) if context_parts else "No specific context provided."

        prompt = f"""Context:
                     {context}
                     
                     Question: {query}
                     
                     Answer (based on the context above):"""

        try:
            answer = self._generate(
                prompt,
                max_tokens=max_length,
                temperature=0.4,
                system_message=system_message
            )
            return answer.strip()
        except Exception as e:
            logger.error("❌ Answer generation failed: %s", e)
            return "I'm sorry, I couldn't generate an answer to your question."

    def process_text_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Complete pipeline for processing text: extract relations, entities, keywords.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing all extracted information
        """
        logger.info("🔄 Running complete text processing pipeline")

        # Extract relations
        relations = self.extract_relations(text)

        # Extract entities from relations
        entities = []
        entity_texts = set()

        for relation in relations:
            for entity_text in [relation['source'], relation['target']]:
                if entity_text not in entity_texts:
                    entity_texts.add(entity_text)
                    entities.append({
                        'text': entity_text,
                        'type': 'entity',
                        'description': self.describe_entity(entity_text)
                    })

        # Add descriptions to relations
        for relation in relations:
            relation['description'] = self.describe_relation(
                relation['source'],
                relation['type'],
                relation['target']
            )

        # Extract keywords
        local_keywords, global_keywords = self.extract_keywords(text)

        result = {
            'entities': entities,
            'relations': relations,
            'local_keywords': local_keywords,
            'global_keywords': global_keywords,
            'text': text
        }

        logger.info("✅ Pipeline completed: %d entities, %d relations, %d keywords",
                   len(entities), len(relations), len(local_keywords) + len(global_keywords))

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        try:
            model_info = ollama.show(self.model_name)
            return {
                "model_name": self.model_name,
                "provider": "Ollama",
                "host": self.host,
                "port": self.port,
                "model_info": model_info,
                "capabilities": {
                    "relation_extraction": True,
                    "entity_description": True,
                    "keyword_extraction": True,
                    "answer_generation": True,
                    "text_pipeline": True
                }
            }
        except Exception as e:
            logger.error("❌ Failed to get model info: %s", e)
            return {"error": str(e)}

    def clear_resources(self):
        """Clean up resources."""
        logger.debug("🧹 Clearing resources")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.clear_resources()
        if exc_type is not None:
            logger.error("⚠️ Context exited with exception: %s", exc_val)
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test the complete pipeline
    try:
        # Initialize with a popular model
        with OllamaLLM(model_name="gemma3:1b") as llm:

            # pylint: disable=invalid-name
            # Sample text for testing
            sample_text = """
            Vitamin D is a crucial nutrient that regulates calcium absorption in the intestines. 
            This process is essential for maintaining bone health and preventing osteoporosis. 
            Calcium, when properly absorbed, contributes to bone mineralization and overall skeletal strength. 
            Research studies have shown that Vitamin D deficiency is directly linked to increased risk of fractures. 
            The sun exposure helps the body produce Vitamin D naturally through skin synthesis.
            """

            print("🔬 Testing Complete OllamaLLM Pipeline")
            print("=" * 50)

            # Test complete pipeline
            print("\n🔄 Running Complete Pipeline")
            result = llm.process_text_pipeline(sample_text)

            print("\n📊 Pipeline Results:")
            print(f"- Entities: {len(result['entities'])}")
            print(f"- Relations: {len(result['relations'])}")
            print(f"- Local Keywords: {len(result['local_keywords'])}")
            print(f"- Global Keywords: {len(result['global_keywords'])}")

            # Display detailed results
            print("\n🏷️ Entities:")
            for entity in result['entities']:
                print(f"  • {entity['text']}: {entity['description']}")

            print("\n🔗 Relations:")
            for relation in result['relations']:
                print(f"  • {relation['source']} --{relation['type']}--> {relation['target']}")
                print(f"    Description: {relation['description']}")

            print(f"\n🔑 Local Keywords: {', '.join(result['local_keywords'])}")
            print(f"🌐 Global Keywords: {', '.join(result['global_keywords'])}")

            # Test answer generation
            print("\n💬 Testing Answer Generation")
            query = "How does Vitamin D affect bone health?"
            answer = llm.generate_answer(
                query,
                result['entities'],
                result['relations']
            )
            print(f"Query: {query}")
            print(f"Answer: {answer}")

            # Display model info
            print("\n📋 Model Information:")
            model_info = llm.get_model_info()
            print(f"Model: {model_info['model_name']}")
            print(f"Provider: {model_info['provider']}")
            print(f"Server: {model_info['host']}:{model_info['port']}")

    except Exception as e:
        logger.error("❌ Test failed: %s", e)
