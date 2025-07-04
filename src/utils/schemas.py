"""
Pydantic models for entity and relationship description requests.

This module provides data validation models for handling entity descriptions
and relationship mappings in knowledge graph or NLP processing systems.
"""

from pydantic import BaseModel, Field


class EntityDescriptionRequest(BaseModel):
    """
    Request model for entity description operations.
    
    This model represents a request to describe or process an entity with
    specific text content and categorization type.
    
    Attributes:
        text: The textual content describing the entity
        type: The classification or category type of the entity
    
    Example:
        >>> request = EntityDescriptionRequest(
        ...     text="A software engineer with 5 years of experience",
        ...     type="person"
        ... )
    """

    text: str = Field(
        ...,
        description="The descriptive text content of the entity",
        min_length=1,
        max_length=10000
    )
    type: str = Field(
        ...,
        description="The classification type or category of the entity",
        min_length=1,
        max_length=100
    )


class RelationDescriptionRequest(BaseModel):
    """
    Request model for relationship description operations.
    
    This model represents a request to describe or process a relationship
    between two entities with a specified relationship type.
    
    Attributes:
        source: The source entity identifier in the relationship
        type: The type or nature of the relationship
        target: The target entity identifier in the relationship
    
    Example:
        >>> request = RelationDescriptionRequest(
        ...     source="John Doe",
        ...     type="works_for",
        ...     target="TechCorp Inc"
        ... )
    """

    source: str = Field(
        ...,
        description="The identifier of the source entity in the relationship",
        min_length=1,
        max_length=500
    )
    type: str = Field(
        ...,
        description="The type or category of the relationship",
        min_length=1,
        max_length=100
    )
    target: str = Field(
        ...,
        description="The identifier of the target entity in the relationship",
        min_length=1,
        max_length=500
    )
