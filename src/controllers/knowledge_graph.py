"""
Knowledge Graph Construction and Management System

This module provides a complete knowledge graph implementation with:
- Entity and relation storage with embeddings
- FAISS-based similarity search
- Graph traversal capabilities

Key Components:
- KnowledgeGraph: Core class for graph operations
- Entity/Relation: Data structures for graph elements
- FAISS: For efficient vector similarity search

Typical Workflow:
1. Initialize KnowledgeGraph with embedding model
2. Add entities and relations
3. Query the graph for neighbors/relations
4. Use in downstream reasoning tasks
"""

# pylint: disable=redefined-outer-name
import logging
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Setup main path for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

#pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.schemas import Entity, Relation

logger = setup_logging()

class KnowledgeGraph:
    """
    A knowledge graph implementation with semantic search capabilities.
    
    Features:
    - Stores entities and relations with vector embeddings
    - FAISS-based similarity search for entities/relations
    - Metadata management for graph elements
    - Nearest neighbor queries
    
    Construction:
    >>> kg = KnowledgeGraph(embedding_model=embedding_model)
    >>> kg.add_entity(entity)
    >>> kg.add_relation(relation)
    
    Querying:
    >>> neighbors = kg.get_entity_neighbors(entity_id)
    """

    def __init__(self, embedding_dim: int = 384,
                 embedding_model: SentenceTransformer = None):
        """
        Initialize the knowledge graph storage and indexes.
        
        Args:
            embedding_dim: Dimension of embeddings (default: 384)
            embedding_model: SentenceTransformer model for embeddings
        """
        logger.info("Initializing KnowledgeGraph with embedding_dim=%d", embedding_dim)
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Optional[Dict] = None
        self.relation_index: Optional[Dict] = None
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        try:
            self._initialize_indexes()
            logger.info("KnowledgeGraph initialized successfully")
        except Exception as e:
            logger.critical("Failed to initialize KnowledgeGraph: %s", str(e))
            raise

    def _initialize_indexes(self):
        """Initialize FAISS indexes for efficient similarity search."""
        logger.debug("Initializing FAISS indexes")
        try:
            self.entity_index = {
                "id_to_entity": {},
                "faiss_index": faiss.IndexFlatL2(self.embedding_dim)
            }
            self.relation_index = {
                "id_to_relation": {},
                "faiss_index": faiss.IndexFlatL2(self.embedding_dim)
            }
            logger.debug("FAISS indexes initialized")
        except Exception as e:
            logger.error("Index initialization failed: %s", str(e))
            raise

    def add_entity(self, entity: Entity, text: Optional[str] = None):
        """
        Add an entity to the knowledge graph with embedding.
        """
        logger.info("Adding entity: %s", entity.id)
        try:
            if entity.id in self.entities:
                logger.warning("Entity %s exists - merging metadata", entity.id)
                existing = self.entities[entity.id]
                existing.metadata.update(entity.metadata or {})
                if entity.description and not existing.description:
                    existing.description = entity.description
                return

            # Generate and store embedding
            embed_text = entity.description or text or entity.name
            logger.debug("Generating embedding for entity %s", entity.id)
            embedding = self.embedding_model.encode(embed_text)

            # Store embedding in the entity
            entity.embedding = embedding  # This requires the Entity class to support this

            self.entities[entity.id] = entity
            idx = len(self.entity_index["id_to_entity"])
            self.entity_index["id_to_entity"][entity.id] = idx
            self.entity_index["faiss_index"].add(np.array([embedding]))

            logger.debug("Added entity %s to index", entity.id)
        except Exception as e:
            logger.error("Failed to add entity %s: %s", entity.id, str(e))
            raise

    def add_relation(self, relation: Relation, text: Optional[str] = None):
        """
        Add a relation between entities with embedding.
        
        Args:
            relation: Relation object to add
            text: Optional context text for embedding
            
        Raises:
            ValueError: If entities don't exist
            RuntimeError: If embedding fails
        """
        logger.info("Adding relation %s between %s and %s",
                   relation.type, relation.source_entity_id, relation.target_entity_id)
        try:
            if relation.source_entity_id not in self.entities:
                raise ValueError(f"Source entity {relation.source_entity_id} not found")
            if relation.target_entity_id not in self.entities:
                raise ValueError(f"Target entity {relation.target_entity_id} not found")

            self.relations.append(relation)
            embed_text = relation.description or relation.type
            if text:
                embed_text = f"{embed_text} {text}" if embed_text else text

            logger.debug("Generating embedding for relation %s", relation.id)
            embedding = self.embedding_model.encode(embed_text)
            idx = len(self.relation_index["id_to_relation"])
            self.relation_index["id_to_relation"][relation.id] = idx
            self.relation_index["faiss_index"].add(np.array([embedding]))

            logger.debug("Added relation %s to index", relation.id)
        except Exception as e:
            logger.error("Failed to add relation %s: %s", relation.id, str(e))
            raise

    def get_entity_neighbors(self, entity_id: str, k: int = 5) -> List[Entity]:
        """
        Find semantically similar entities.
        
        Args:
            entity_id: Entity to find neighbors for
            k: Number of neighbors to return
            
        Returns:
            List of similar entities
            
        Raises:
            ValueError: If entity not found
            RuntimeError: If search fails
        """
        logger.info("Finding neighbors for entity %s", entity_id)
        try:
            if entity_id not in self.entities:
                raise ValueError(f"Entity {entity_id} not found")

            idx = self.entity_index["id_to_entity"][entity_id]
            embedding = self.entity_index["faiss_index"].reconstruct(idx)

            _, indices = self.entity_index["faiss_index"].search(
                np.array([embedding]), k+1
            )

            neighbors = []
            for i in indices[0]:
                if i != idx and i in self.entity_index["id_to_entity"]:
                    neighbor_id = self.entity_index["id_to_entity"][i]
                    neighbors.append(self.entities[neighbor_id])
                if len(neighbors) >= k:
                    break

            logger.debug("Found %d neighbors for %s", len(neighbors), entity_id)
            return neighbors

        except Exception as e:
            logger.error("Neighbor search failed for %s: %s", entity_id, str(e))
            raise

    def search_entities_by_embedding(self, query_embedding, top_k=3):
        """
        Search entities by embedding similarity using cosine similarity.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matching entities sorted by similarity
        """
        try:
            results = []
            query_embedding = np.array(query_embedding)

            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            for entity in self.entities.values():
                if not hasattr(entity, 'embedding') or entity.embedding is None:
                    continue

                # Normalize entity embedding
                entity_embedding = np.array(entity.embedding)
                entity_embedding = entity_embedding / np.linalg.norm(entity_embedding)

                # Calculate cosine similarity
                score = float(query_embedding @ entity_embedding)
                results.append((score, entity))

            # Sort by score (descending) and return top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [entity for score, entity in results[:top_k]]

        except Exception as e:
            logger.error("Embedding search failed: %s", str(e))
            raise
