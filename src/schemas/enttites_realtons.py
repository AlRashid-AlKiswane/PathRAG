"""
Pydantic models for entity and relationship description requests.

This module provides data validation models for handling entity descriptions
and relationship mappings in knowledge graph or NLP processing systems.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'metadata': self.metadata,
            'embedding': self.embedding
        }


@dataclass
class Relation:
    """Represents a relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
