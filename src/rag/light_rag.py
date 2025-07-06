"""
LightRAG - Lightweight Retrieval-Augmented Generation System with Knowledge Graph

A modular system for building and querying a knowledge graph from documents, combining:
- Entity and relation extraction
- Semantic embedding generation
- Dual-level (specific + abstract) retrieval
- LLM-augmented response generation
"""

# pylint: disable=redefined-outer-name
import logging
import os
import sys
from typing import List, Dict, Union,Tuple, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable= 
from src.utils import setup_logging
from src.controllers import KnowledgeGraph, DocumentProcessor
from src.llmsprovider import NERModel
from src.schemas import Entity, Relation
from src.llmsprovider import OllamaModel

logger = setup_logging()

class LightRAG:
    """
    Main class for the LightRAG system implementing RAG with knowledge graph.
    
    Features:
    - Document ingestion and processing pipeline
    - Knowledge graph construction
    - Hybrid retrieval capabilities
    - Response generation
    
    Example Usage:
        >>> rag = LightRAG()
        >>> rag.ingest_document("document.txt")
        >>> response = rag.query("What is the main topic?")
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the LightRAG system.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        logger.info("Initializing LightRAG system")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.ner_model = NERModel()
            self.ollama_model: OllamaModel = OllamaModel()
            self.document_processor = DocumentProcessor(embedding_model=self.embedding_model)
            self.knowledge_graph = KnowledgeGraph(embedding_model=self.embedding_model)
            self.chunk_store = []
            logger.info("LightRAG initialized successfully")
        except Exception as e:
            logger.critical("Failed to initialize LightRAG: %s", str(e))
            raise

    def ingest_document(self, document_source: Union[str, Path]) -> Tuple[int, int]:
        """
        Process and ingest a document into the system.
        
        Args:
            document_source: Path to document or raw text
            
        Returns:
            Tuple of (entities_added, relations_added)
            
        Raises:
            ValueError: For invalid input
            IOError: For file reading issues
        """
        logger.info("Starting document ingestion")
        try:
            # Step 1: Process document into chunks
            if isinstance(document_source, (str, Path)) and os.path.isfile(document_source):
                logger.debug("Processing file: %s", document_source)
                with open(document_source, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.debug("Processing raw text input")
                text = document_source
                
            chunks = self.document_processor.process(text)
            self.chunk_store.extend(chunks)
            
            # Step 2: Extract entities and relations
            entities, relations = self._extract_entities_and_relations(chunks)
            
            # Step 3: Add to knowledge graph
            for entity in entities:
                self.knowledge_graph.add_entity(entity)
            for relation in relations:
                self.knowledge_graph.add_relation(relation)
                
            logger.info("Document ingested. Added %d entities and %d relations", 
                       len(entities), len(relations))
            return len(entities), len(relations)
            
        except Exception as e:
            logger.error("Document ingestion failed: %s", str(e))
            raise

    def _extract_entities_and_relations(self, chunks: List[Dict]) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from processed chunks using NER model.
        
        Args:
            chunks: List of processed document chunks with text and metadata
            
        Returns:
            Tuple of (entities, relations) where:
            - entities: List of extracted Entity objects
            - relations: List of extracted Relation objects
            
        Raises:
            RuntimeError: If entity extraction fails
        """
        logger.info("Extracting entities and relations from %d chunks", len(chunks))
        entities = []
        relations = []
        seen_entities = set()  # For entity deduplication
        
        try:
            for chunk in chunks:
                text = chunk['text']
                chunk_id = chunk.get('metadata', {}).get('chunk_id', str(len(entities)))
                
                # Extract entities using NER model
                try:
                    ner_results = self.ner_model.predict(text=text)
                    for ent in ner_results:
                        # Create normalized entity ID
                        entity_text = ent["word"].lower().replace(" ", "_")
                        entity_id = f"{entity_text}_{ent['start']}"
                        
                        if entity_id not in seen_entities:
                            entities.append(Entity(
                                id=entity_id,
                                name=ent["word"],
                                type=ent["entity_group"],
                                description=text[max(0, ent['start']-50):ent['end']+50],
                                metadata={
                                    "source_text": text,
                                    "start_pos": ent['start'],
                                    "end_pos": ent['end'],
                                    "confidence": float(ent['score']),
                                    "chunk_id": chunk_id
                                }
                            ))
                            seen_entities.add(entity_id)
                            logger.debug("Extracted entity: %s (%s)", ent["word"], ent["entity_group"])
                except Exception as e:
                    logger.warning("NER failed for chunk %s: %s", chunk_id, str(e))
                    continue
                
                # Extract relations (simplified example - replace with actual relation extraction)
                if len(entities) >= 2:
                    last_two = entities[-2:]
                    relations.append(Relation(
                        id=f"rel_{len(relations)}",
                        type="associated_with",
                        source_entity_id=last_two[0].id,
                        target_entity_id=last_two[1].id,
                        description=f"Co-occurrence in chunk {chunk_id}",
                        metadata={
                            "chunk_id": chunk_id,
                            "extraction_method": "co-occurrence",
                            "confidence": 0.7  # Example confidence score
                        }
                    ))
                    logger.debug("Created relation between %s and %s", 
                            last_two[0].name, last_two[1].name)
            
            logger.info("Extracted %d entities and %d relations", len(entities), len(relations))
            return entities, relations
            
        except Exception as e:
            logger.error("Entity/relation extraction failed: %s", str(e))
            raise RuntimeError("Failed to extract entities and relations") from e

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Query the knowledge graph and generate a response.
        
        Args:
            question: The query string
            top_k: Number of top documents/entities to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info("Processing query: %s", question)
        try:
            # Step 1: Embed the question
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)

            # Step 2: Specific-level retrieval - Retrieve relevant chunks
            logger.debug("Retrieving top-%d chunks", top_k)
            chunk_scores = []
            for chunk in self.chunk_store:
                chunk_embedding = chunk['embedding']
                score = float(question_embedding @ chunk_embedding)
                chunk_scores.append((score, chunk))
            chunk_scores.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for _, chunk in chunk_scores[:top_k]]

            # Step 3: Abstract-level retrieval - Graph-based entity retrieval
            logger.debug("Retrieving top-%d entities from knowledge graph", top_k)
            graph_results = self.knowledge_graph.search_entities_by_embedding(
                question_embedding, top_k=top_k
            )

            # Step 4: Combine and rerank results (simple merge for now)
            retrieved_texts = []
            for chunk in top_chunks:
                retrieved_texts.append(chunk['text'])
            for entity in graph_results:
                retrieved_texts.append(entity.description)

            # Prepare entities and relations for response generation
            entities_serialized = [e() for e in graph_results]
            relations_serialized = self.knowledge_graph.relation_index.values()

            # Step 5: Generate final answer using LLM
            context = "\n".join(retrieved_texts)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            response_text = self._generate_response(
                query=prompt,
                entities=entities_serialized,
                relations=relations_serialized
            )

            logger.debug("Query processed successfully")
            return {
                "question": question,
                "response": response_text.strip(),
                "top_chunks": [c['text'] for c in top_chunks],
                "top_entities": [e.name for e in graph_results]
            }
        except Exception as e:
            logger.error("Query processing failed: %s", str(e))
            raise

    def _generate_response(self, query: str,
                        entities: List[Dict[str, Any]],
                        relations: List[Dict[str, Any]]) -> str:
        """
        Generate a final response from the LLM using query, entities, and relations.

        Args:
            query: The original user question.
            entities: List of entity dictionaries.
            relations: List of relation dictionaries.

        Returns:
            Generated response string from the LLM.
        """
        prompt = f"QUERY: {query}\nENTITIES: {entities}\nRELATIONS: {relations}"
        try:
            response = self.ollama_model.generate(
                query=query,
                entities=entities,
                relations=relations
            )
            return response
        except Exception as e:
            logger.error("LLM response generation failed: %s", str(e))
            raise RuntimeError("Failed to generate response using Ollama model") from e
