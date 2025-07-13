"""
chatbot_schema.py

Defines the Pydantic data model for chatbot requests in the Graph-RAG API.

This schema is used to validate and document incoming chatbot queries,
ensuring proper formatting and constraints before further processing
by the LLM and retrieval pipeline.

Classes:
    - Chatbot: Validates user input and retrieval configuration parameters
      such as top_k documents, LLM temperature, token limits, retrieval mode, etc.

Example:
    {
        "query": "What are the benefits of Graph Neural Networks?",
        "top_k": 5,
        "temperature": 0.7,
        "max_new_tokens": 256,
        "max_input_tokens": 1024,
        "mode_retrieval": "hybrid",
        "user_id": "user_123",
        "cache": true
    }

Author: AlRashid @TessFlod LLC
Created: 2025-07-13
"""

from pydantic import BaseModel, Field

class Chatbot(BaseModel):
    """
    Request schema for interacting with the chatbot endpoint.

    Attributes:
        query (str): The user's input query.
        top_k (int): Number of top documents to retrieve from the retriever.
        temperature (float): Sampling temperature for the LLM response.
        max_new_tokens (int): Maximum number of tokens to generate in response.
        max_input_tokens (int): Maximum tokens allowed in the prompt input.
        mode_retrieval (str): Retrieval mode, e.g., 'semantic', 'entity', or 'hybrid'.
        user_id (str): Unique identifier for the user session or client.
        cache (bool): Whether to cache the query and response.
    """
    query: str = Field(..., description="User's input question or prompt.")
    top_k: int = Field(5, ge=1, le=50, description="Number of top documents to retrieve.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="LLM sampling temperature.")
    max_new_tokens: int = Field(256, ge=1, le=1024, description="Max tokens to generate.")
    max_input_tokens: int = Field(1024, ge=1, le=4096, description="Max input tokens allowed.")
    mode_retrieval: str = Field("semantic", description="Retrieval strategy: 'semantic', 'entity', or 'hybrid'.")
    user_id: str = Field(..., description="Unique user or session identifier.")
    cache: bool = Field(True, description="Flag to enable or disable caching.")
