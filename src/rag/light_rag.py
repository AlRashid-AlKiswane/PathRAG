"""
LightRAG - Lightweight Retrieval-Augmented Generation System with Knowledge Graph

A modular system for building and querying a knowledge graph from documents, combining:
- Entity and relation extraction
- Semantic embedding generation
- Dual-level (specific + abstract) retrieval
- LLM-augmented response generation
"""

# pylint: disable=redefined-outer-name, wrong-import-position
import logging
import os
import sys
from dataclasses import asdict
from typing import List, Dict, Union, Tuple, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.infra import setup_logging
from src.controllers import KnowledgeGraph
from src.llms_providers import NERModel
from src.schemas import Entity, Relation

logger = setup_logging()


class LightRAG:
    """
    Core class implementing the LightRAG system with:
    - Embedding model
    - Named entity recognition
    - Knowledge graph construction
    - Chunk-based retrieval
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the LightRAG system.

        Args:
            embedding_model_name: SentenceTransformer model to use for embeddings
        """
        logger.info("Initializing LightRAG system")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.ner_model = NERModel()
            self.knowledge_graph = KnowledgeGraph(embedding_model=self.embedding_model)
            self.chunk_store: List[Dict[str, Any]] = []
            logger.info("LightRAG initialized successfully")
        except Exception as e:
            logger.critical("Failed to initialize LightRAG: %s", str(e))
            raise

    def extract_entities_and_relations(self, chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from processed chunks using NER.

        Args:
            chunks: List of processed document chunks with text and metadata

        Returns:
            Tuple of (entities, relations)
        """
        logger.info("Extracting entities and relations from %d chunks", len(chunks))

        entities: List[Entity] = []
        relations: List[Relation] = []
        seen_entities = set()

        try:
            for chunk in chunks:
                text = chunk['text']
                chunk_id = chunk.get('metadata', {}).get('chunk_id', str(len(entities)))

                try:
                    ner_results = self.ner_model.predict(text=text)
                    for ent in ner_results.get("entities", []):
                        entity_id = f"{ent['text'].lower().replace(' ', '_')}_{ent['start']}"
                        if entity_id in seen_entities:
                            continue
                        entity = Entity(
                            id=entity_id,
                            name=ent["text"],
                            type=ent["type"],
                            description=text[max(0, ent['start'] - 50):ent['end'] + 50],
                            metadata={
                                "source_text": text,
                                "start_pos": ent['start'],
                                "end_pos": ent['end'],
                                "confidence": float(ent['score']),
                                "chunk_id": chunk_id
                            }
                        )
                        entities.append(entity)
                        seen_entities.add(entity_id)
                        logger.debug("Extracted entity: %s (%s)", ent["text"], ent["type"])

                except Exception as e:
                    logger.warning("NER failed for chunk %s: %s", chunk_id, str(e))
                    continue

                # Simplified relation extraction
                if len(entities) >= 2:
                    e1, e2 = entities[-2:]
                    relations.append(Relation(
                        id=f"rel_{len(relations)}",
                        type="associated_with",
                        source_entity_id=e1.id,
                        target_entity_id=e2.id,
                        description=f"Co-occurrence in chunk {chunk_id}",
                        metadata={
                            "chunk_id": chunk_id,
                            "extraction_method": "co-occurrence",
                            "confidence": 0.7
                        }
                    ))
                    logger.debug("Created relation between %s and %s", e1.name, e2.name)

            # Add to graph
            for entity in entities:
                self.knowledge_graph.add_entity(entity)
            for relation in relations:
                self.knowledge_graph.add_relation(relation)

            logger.info(
                "Extraction complete: %d entities, %d relations", len(entities), len(relations))
            return entities, relations

        except Exception as e:
            logger.error("Entity/relation extraction failed: %s", str(e))
            raise RuntimeError("Failed to extract entities and relations") from e

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve relevant information from chunks and knowledge graph.

        Args:
            question: The user's query
            top_k: Number of top items to retrieve

        Returns:
            Dictionary with question, top_chunks, top_entities, relations, context, etc.
        """
        logger.info("Processing retrieval query: %s", question)

        try:
            question_emb = self.embedding_model.encode(question, convert_to_tensor=True)

            # Retrieve top chunks
            chunk_scores = [
                (float(question_emb @ chunk['embedding']), chunk)
                for chunk in self.chunk_store
            ]
            chunk_scores.sort(key=lambda x: x[0], reverse=True)
            top_chunks = chunk_scores[:top_k]

            # Retrieve entities
            top_entities = self.knowledge_graph.search_entities_by_embedding(
                question_emb, top_k=top_k
                )

            # Retrieve relations linked to those entities
            relevant_relations = [
                r for r in self.knowledge_graph.relations
                if any(e.id in (r.source_entity_id, r.target_entity_id) for e in top_entities)
            ]

            combined_context = "\n".join(
                [c['text'] for _, c in top_chunks] +
                [e.description for e in top_entities if e.description]
            )

            return {
                "question": question,
                "top_chunks": [
                    {
                        "text": c['text'],
                        "score": s,
                        "metadata": c.get('metadata', {})
                    } for s, c in top_chunks
                ],
                "top_entities": [asdict(e) for e in top_entities],
                "relations": [asdict(r) for r in relevant_relations],
                "combined_context": combined_context,
                "retrieval_success": True
            }

        except Exception as e:
            logger.error("Retrieval failed: %s", str(e))
            return {
                "question": question,
                "error": str(e),
                "retrieval_success": False
            }
