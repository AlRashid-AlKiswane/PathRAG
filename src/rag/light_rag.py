"""
LightRAG - Lightweight Retrieval-Augmented Generation System with Knowledge Graph

A modular system for building and querying a knowledge graph from documents, combining:
- Entity and relation extraction
- Semantic embedding generation
- Dual-level (specific + abstract) retrieval
- LLM-augmented response generation

Key Features:
1. Document Processing Pipeline:
   - Chunking with semantic overlap
   - Joint entity and relation extraction
   - Entity/relation profiling with LLM enrichment
   - Automatic deduplication and graph merging

2. Knowledge Graph Components:
   - Entity and relation embeddings
   - Dual indexes (textual + semantic)
   - One-hop neighborhood expansion
   - Persistent storage support

3. Query Capabilities:
   - Hybrid keyword/vector retrieval
   - Context-aware response generation
   - Automatic relevance feedback via graph traversal

Usage:
>>> rag = LightRAG(embedding_model, ner_model, llm_model)
>>> rag.process_document("Your text here...")
>>> response = rag.query("Your question?")

Designed for efficiency in both construction and query phases while maintaining
interpretability through explicit knowledge graph representation.
"""
import logging
import json
import hashlib
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import setup_logging
logger = setup_logging()

class LightRAG:
    """
    Lightweight Retrieval-Augmented Generation system with knowledge graph capabilities.
    Handles document processing, entity/relation extraction, and dual-level retrieval.
    """
    def __init__(self, embedding_model, ner_model, llm_model):
        """
        Initialize the LightRAG system with required models.
        
        Args:
            embedding_model: Model for generating embeddings
            ner_model: Named Entity Recognition model
            llm_model: Large Language Model for text generation and refinement
        """
        try:
            self.embedding_model = embedding_model
            self.ner_model = ner_model
            self.llm = llm_model

            # Initialize knowledge graph structure
            self.knowledge_graph = {
                "entities": {},
                "relations": {},
                "entity_index": defaultdict(set),
                "relation_index": defaultdict(set),
                "entity_embeddings": {},
                "relation_embeddings": {}
            }

            # Cache for frequently accessed data
            self.cache = {
                "entity_texts": {},
                "relation_texts": {}
            }

            logger.info("LightRAG initialized successfully")
        except Exception as e:
            logger.error("Error initializing LightRAG: %s}", str(e))
            raise

    def process_document(self, document: str,
                        chunk_size: int = 1000,
                        chunk_overlap: int = 50) -> None:
        """
        Process a document through the full pipeline:
        1. Chunking
        2. Entity and relation extraction
        3. Profiling and key-value generation
        4. Deduplication and graph merging

        Args:
            document: Input text document to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        try:
            logger.info("Processing document (size: %d chars)", len(document))

            chunks = self._chunk_document(
                text=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            logger.info("Document split into %d chunks", len(chunks))

            for i, chunk in enumerate(chunks):
                try:
                    logger.debug("Processing chunk %d/%d", i + 1, len(chunks))

                    # Extract entities & relations
                    entities, relations = self._extract_entities_relations(chunk)

                    # Profile and enrich
                    profiled_entities = self._profile_entities(entities)
                    profiled_relations = self._profile_relations(relations)

                    # Generate embeddings
                    self._generate_embeddings(profiled_entities, profiled_relations)

                    # Merge into knowledge graph
                    self._merge_into_graph(profiled_entities, profiled_relations)

                    logger.debug("Chunk %d processed successfully", i + 1)
                except (ValueError, TypeError) as e:
                    logger.error("Specific error occurred: %s", str(e))
                    continue

            logger.info("Document processing completed")
        except Exception as e:
            logger.error("Error in document processing: %s", str(e))
            raise


    def _chunk_document(self, text: str, chunk_size: int, chunk_overlap: int = 50) -> List[str]:
        """Split document into semantically meaningful chunks using LangChain with overlap."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.debug("Split document into %d chunks", len(chunks))
            return chunks
        except (ValueError, TypeError) as e:
            logger.error("Invalid input for chunking: %s", str(e))
            raise
        except Exception as e:
            logger.exception("Unexpected error in document chunking - Error: %s", str(e))
            raise

    def _extract_entities_relations(self, text: str) -> Tuple[List[dict], List[dict]]:
        """
        Extract entities and relations using NER and relation extraction.
        Combines model-based extraction with LLM refinement.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (entities, relations) where each is a list of dictionaries
        """
        try:
            logger.debug("Extracting entities and relations")

            # Step 1: Use NER model to identify entities
            ner_results = self.ner_model.predict(text)
            entities = [
                {
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "source_text": text
                }
                for ent in ner_results.get("entities", [])
            ]

            logger.debug("Found %d entities", len(entities))

            # Step 2: Use LLM to identify relationships between entities
            relation_prompt = self._build_relation_prompt(text=text, entities=entities)
            llm_relations = self.llm.extract_relations(relation_prompt)

            # Standardize relation format
            relations = []
            for rel in llm_relations:
                source = next((e for e in entities if e["text"] == rel["source"]), None)
                target = next((e for e in entities if e["text"] == rel["target"]), None)

                if source and target:
                    relations.append({
                        'source': source['text'],
                        'target': target['text'],
                        'type': rel['type'],
                        'source_text': text,
                        'metadata': {
                            'source_type': source['type'],
                            'target_type': target['type']
                        }
                    })

            logger.debug("Found %d relations", len(relations))
            return entities, relations
        except Exception as e:
            logger.error("Error in entity/relation extraction: %s", str(e) )
            raise

    def _build_relation_prompt(self, text: str, entities: List[dict]) -> str:
        """Construct prompt for relation extraction."""
        try:
            entity_list = "\n".join([f"- {e['text']} ({e['type']})" for e in entities])
            prompt = f"""
            Text: {text}
            
            Identified entities:
            {entity_list}
            
            Identify relationships between these entities in the format:
            - [Entity1] [Relationship] [Entity2]
            For example:
            - Cardiologist diagnoses Heart Disease
            """
            return prompt
        except Exception as e:
            logger.error("Error building relation prompt: %s", str(e))
            raise

    def _profile_entities(self, entities: List[dict]) -> List[dict]:
        """
        Generate enriched profile for entities.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of enriched entity profiles
        """
        processed = []
        try:
            for entity in entities:
                # Generate unique ID
                entity_id = self._generate_id(f"{entity['text']}_{entity['type']}")

                # Get additional info from LLM
                description = self.llm.describe_entity(entity['text'], entity['type'])

                processed.append({
                    'id': entity_id,
                    'text': entity['text'],
                    'type': entity['type'],
                    'description': description,
                    'source_text': entity['source_text'],
                    'metadata': entity.get('metadata', {})
                })

            logger.debug("Profiled %d entities", len(processed))
            return processed
        except Exception as e:
            logger.error("Error profiling entities: %s", str(e))
            raise

    def _profile_relations(self, relations: List[dict]) -> List[dict]:
        """
        Generate enriched profile for relations.
        
        Args:
            relations: List of relation dictionaries
            
        Returns:
            List of enriched relation profiles
        """
        processed = []
        try:
            for rel in relations:
                # Generate unique ID
                rel_id = self._generate_id(f"{rel['source']}_{rel['type']}_{rel['target']}")

                # Get additional info from LLM
                description = self.llm.describe_relation(
                    rel["source"], rel["type"], rel["target"]
                )

                processed.append({
                    'id': rel_id,
                    'source': rel['source'],
                    'target': rel['target'],
                    'type': rel['type'],
                    'description': description,
                    'source_text': rel['source_text'],
                    'metadata': rel.get('metadata', {})
                })

            logger.debug("Profiled %d relations", len(processed))
            return processed
        except Exception as e:
            logger.error("Error profiling relations: %s", str(e))
            raise

    def _generate_embeddings(self, entities: List[dict], relations: List[dict]) -> None:
        """
        Generate embeddings for entities and relations.
        
        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries
        """
        try:
            # Entity embeddings based on their text and type
            entity_texts = [f"{e['text']} {e['type']}" for e in entities]
            if entity_texts:
                entity_embeddings = self.embedding_model.encode(entity_texts)
                for e, emb in zip(entities, entity_embeddings):
                    self.knowledge_graph["entity_embeddings"][e["id"]] = emb
                    self.cache["entity_texts"][e["id"]] = entity_texts[entities.index(e)]

            # Relation embeddings based on their type and connected entities
            relation_texts = [
                f"{r['source']} {r['type']} {r['target']}" for r in relations
            ]

            if relation_texts:
                relation_embeddings = self.embedding_model.encode(relation_texts)
                for r, emb in zip(relations, relation_embeddings):
                    self.knowledge_graph["relation_embeddings"][r["id"]] = emb
                    self.cache["relation_texts"][r["id"]] = relation_texts[relations.index(r)]

            logger.debug("Generated embeddings for entities and relations")
        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e))
            raise

    def _merge_into_graph(self, entities: List[dict], relations: List[dict]) -> None:
        """
        Merge new entities and relations into the knowledge graph.
        
        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries
        """
        try:
            # Merge entities
            for entity in entities:
                entity_id = entity["id"]

                if entity_id in self.knowledge_graph["entities"]:
                    # Update existing entity
                    existing = self.knowledge_graph["entities"][entity_id]
                    existing["source_text"] += f"\n\n{entity['source_text']}"
                    if "description" in entity:
                        existing["description"] += f"\n\n{entity['description']}"
                else:
                    # Add new entity
                    self.knowledge_graph['entities'][entity_id] = entity
                    self._add_to_index('entity', entity['text'], entity_id)
                    if 'type' in entity:
                        self._add_to_index('entity', entity['type'], entity_id)

            # Merge relations
            for relation in relations:
                rel_id = relation['id']

                if rel_id in self.knowledge_graph['relations']:
                    # Update existing relation
                    existing = self.knowledge_graph['relations'][rel_id]
                    existing['source_text'] += f"\n\n{relation['source_text']}"
                    if 'description' in relation:
                        existing['description'] += f"\n\n{relation['description']}"
                else:
                    # Add new relation
                    self.knowledge_graph['relations'][rel_id] = relation
                    self._add_to_index('relation', relation['type'], rel_id)
                    # Add variants to index
                    self._add_to_index('relation',
                                      f"{relation['source']} {relation['type']}",
                                      rel_id)
                    self._add_to_index('relation',
                                      f"{relation['type']} {relation['target']}",
                                      rel_id)

            logger.debug("Merged entities and relations into knowledge graph")
        except Exception as e:
            logger.error("Error merging into graph: %s", str(e))
            raise

    def _add_to_index(self, index_type: str, key: str, item_id: str) -> None:
        """
        Add an item to the appropriate index.
        
        Args:
            index_type: Type of index ('entity' or 'relation')
            key: Index key
            item_id: ID of the item to index
        """
        try:
            index = self.knowledge_graph[f"{index_type}_index"]
            index[key].add(item_id)
        except Exception as e:
            logger.error("Error adding to index: %s", str(e))
            raise

    def _generate_id(self, text: str) -> str:
        """
        Generate a consistent hash-based ID.
        
        Args:
            text: Input text to hash
            
        Returns:
            SHA256 hash of the input text
        """
        try:
            return hashlib.sha256(text.encode()).hexdigest()
        except Exception as e:
            logger.error("Error generating ID: %s", str(e))
            raise

    def query(self, query_text: str, top_k: int = 5) -> str:
        """
        Execute a dual-level retrieval and generate an answer.
        
        Args:
            query_text: The query to process
            top_k: Number of top items to retrieve at each level

        Returns:
            Generated answer combining retrieved information
        """
        try:
            logger.info("Processing query: %s", query_text)

            # Step 1: Extract keywords and generate query embedding
            local_keywords, global_keywords = self.llm.extract_keywords(query_text)
            query_embedding = self.embedding_model.encode(query_text)

            # Step 2: Dual-level retrieval
            retrieved = self._dual_level_retrieval(
                query_embedding=query_embedding,
                local_keywords=local_keywords,
                global_keywords=global_keywords,
                top_k=top_k
            )

            # Step 3: Generate response
            response = self.llm.generate_answer(
                query=query_text,
                entities=retrieved['entities'],
                relations=retrieved['relations']
            )

            logger.info("Query processed successfully")
            return response
        except Exception as e:
            logger.error("Error processing query: %s", str(e))
            raise

    def _dual_level_retrieval(self, query_embedding: np.ndarray,
                            local_keywords: List[str],
                            global_keywords: List[str],
                            top_k: int) -> dict:
        """
        Perform dual-level (specific + abstract) retrieval.

        Args:
            query_embedding: Embedding of the query
            local_keywords: Keywords for specific retrieval
            global_keywords: Keywords for abstract retrieval
            top_k: Number of items to retrieve

        Returns:
            Dictionary containing retrieved entities and relations
        """
        try:
            results = {
                "entities": set(),
                "relations": set()
            }

            # Low-level (specific) retrieval
            entity_results = self._retrieve_by_keywords("entity", local_keywords)
            entity_results.update(self._retrieve_by_embedding(
                query_embedding=query_embedding,
                embedding_dict=self.knowledge_graph["entity_embeddings"],
                top_k=top_k
            ))
            results["entities"].update(entity_results)

            # High-level (abstract) retrieval
            relation_results = self._retrieve_by_keywords("relation", global_keywords)
            relation_results.update(self._retrieve_by_embedding(
                query_embedding=query_embedding,
                embedding_dict=self.knowledge_graph["relation_embeddings"],
                top_k=top_k
            ))
            results["relations"].update(relation_results)

            # Get one-hop neighbors for context expansion
            neighbors = self._get_one_hop_neighbors(results["entities"], results["relations"])
            results["entities"].update(neighbors["entities"])
            results["relations"].update(neighbors["relations"])

            # Convert to full data
            return {
                'entities': [
                    self.knowledge_graph[
                        'entities'
                        ][eid] for eid in results['entities']],
                'relations': [self.knowledge_graph[
                    'relations'
                    ][rid] for rid in results['relations']]
            }
        except Exception as e:
            logger.error("Error in dual-level retrieval: %s", str(e))
            raise


    def _retrieve_by_keywords(self, index_type: str, keywords: List[str]) -> Set[str]:
        """
        Retrieve items matching any of the keywords.

        Args:
            index_type: Type of index to search ('entity' or 'relation')
            keywords: List of keywords to search for

        Returns:
            Set of matching item IDs
        """
        try:
            results = set()
            index = self.knowledge_graph[f"{index_type}_index"]

            for keyword in keywords:
                if keyword in index:
                    results.update(index[keyword])

            return results
        except Exception as e:
            logger.error("Error in keyword retrieval: %s", str(e))
            raise

    def _retrieve_by_embedding(self,
                            query_embedding: np.ndarray,
                            embedding_dict: Dict[str, np.ndarray],
                            top_k: int) -> Set[str]:
        """
        Retrieve top_k most similar items based on embedding similarity using FAISS.
        
        Performs efficient nearest neighbor search with cosine similarity by:
        1. Normalizing embeddings and query vector
        2. Building an in-memory FAISS index
        3. Searching for top_k most similar items
        
        Args:
            query_embedding: Query embedding vector of shape (d,)
            embedding_dict: Dictionary mapping item IDs to embedding vectors
            top_k: Maximum number of items to retrieve (will return fewer if 
                embedding_dict contains fewer than top_k items)
                
        Returns:
            Set of item IDs for the top_k most similar embeddings
            
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If FAISS operations fail
        """
        try:
            # Validate inputs
            if not embedding_dict:
                logger.debug("Empty embedding_dict provided, returning empty set")
                return set()

            if top_k <= 0:
                logger.warning("top_k must be positive, got %d. Setting to 1", top_k)
                top_k = 1

            # Prepare data structures
            ids = list(embedding_dict.keys())
            actual_top_k = min(top_k, len(ids))
            embeddings = np.array(list(embedding_dict.values()))

            # Validate dimensions
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")

            if query_embedding.shape[0] != embeddings.shape[1]:
                raise ValueError(
                    f"Query dim {query_embedding.shape[0]} != embedding dim {embeddings.shape[1]}"
                )

            # Convert to float32 for FAISS
            embeddings = embeddings.astype("float32")
            query_vector = query_embedding.astype("float32").reshape(1, -1)

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            faiss.normalize_L2(query_vector)

            # Build and search index
            index = faiss.IndexFlatIP(embeddings.shape[1])
            # pylint: disable=no-value-for-parameter
            index.add(embeddings)
            # pylint: enable=no-value-for-parameter

            # pylint: disable=no-value-for-parameter
            index.add(embeddings)
            distances, indices = index.search(query_vector, actual_top_k)
            # pylint: enable=no-value-for-parameter

            # Log search quality metrics
            # pylint: disable=logging-fstring-interpolation
            if logger.isEnabledFor(logging.DEBUG):
                avg_distance = np.mean(distances)
                logger.debug(
                    f"FAISS search completed: top_k={actual_top_k}, "
                    f"avg_similarity={avg_distance:.3f}"
                )

            return {ids[i] for i in indices[0] if i >= 0}  # Filter invalid indices

        except Exception as e:
            logger.error(
                "Embedding retrieval failed for top_k=%d with %d embeddings: %s",
                top_k,
                len(embedding_dict),
                str(e),
                exc_info=True
            )
            raise RuntimeError(f"Embedding retrieval failed: {str(e)}") from e

    def _get_one_hop_neighbors(self, entity_ids: Set[str], relation_ids: Set[str]) -> dict:
        """
        Expand retrieval with one-hop neighbors.

        Args:
            entity_ids: Set of entity IDs
            relation_ids: Set of relation IDs

        Returns:
            Dictionary containing neighbor entities and relations
        """
        try:
            neighbors = {
                "entities": set(),
                "relations": set()
            }

            # Follow relations from entities
            for eid in entity_ids:
                for rid, rel in self.knowledge_graph["relations"].items():
                    if eid in (rel["source"], rel["target"]):
                        neighbors["relations"].add(rid)
                        neighbors["entities"].add(rel["source"])
                        neighbors["entities"].add(rel["target"])

            # Follow entities from relations
            for rid in relation_ids:
                rel = self.knowledge_graph['relations'].get(rid, {})
                if 'source' in rel:
                    neighbors['entities'].add(rel['source'])
                if 'target' in rel:
                    neighbors['entities'].add(rel['target'])

            # Remove originals to avoid duplicates
            neighbors['entities'].difference_update(entity_ids)
            neighbors['relations'].difference_update(relation_ids)

            return neighbors
        except Exception as e:
            logger.error("Error getting one-hop neighbors: %s", str(e))
            raise


    def save(self, filepath: str) -> None:
        """
        Save the knowledge graph to a file.

        Args:
            filepath: Path to save the knowledge graph
        """
        try:
            # Convert sets to lists for JSON serialization
            save_data = {
                'entities': self.knowledge_graph['entities'],
                'relations': self.knowledge_graph['relations'],
                'entity_index': {
                    k: list(v) for k, v in self.knowledge_graph['entity_index'].items()},
                'relation_index': {
                    k: list(v) for k, v in self.knowledge_graph['relation_index'].items()},
                'entity_embeddings': {
                    k: v.tolist() for k, v in self.knowledge_graph['entity_embeddings'].items()},
                'relation_embeddings': {
                    k: v.tolist() for k, v in self.knowledge_graph['relation_embeddings'].items()}
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f)

            logger.info("Knowledge graph saved to %s", filepath)
        except Exception as e:
            logger.error("Error saving knowledge graph: %s", str(e))
            raise


    def load(self, filepath: str) -> None:
        """
        Load a knowledge graph from file.

        Args:
            filepath: Path to load the knowledge graph from
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.knowledge_graph = {
                'entities': data['entities'],
                'relations': data['relations'],
                'entity_index': {
                    k: set(v) for k, v in data['entity_index'].items()},
                'relation_index': {
                    k: set(v) for k, v in data['relation_index'].items()},
                'entity_embeddings': {
                    k: np.array(v) for k, v in data['entity_embeddings'].items()},
                'relation_embeddings': {
                    k: np.array(v) for k, v in data['relation_embeddings'].items()}
            }

            # Rebuild cache
            self.cache['entity_texts'] = {
                eid: f"{e['text']} {e['type']}"
                for eid, e in self.knowledge_graph['entities'].items()
            }
            self.cache['relation_texts'] = {
                rid: f"{r['source']} {r['type']} {r['target']}"
                for rid, r in self.knowledge_graph['relations'].items()
            }

            logger.info("Knowledge graph loaded from %s", filepath)
        except Exception as e:
            logger.error("Error loading knowledge graph: %s", str(e))
            raise
