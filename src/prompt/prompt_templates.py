"""
prompt_templates.py

This module defines prompt templates for interacting with Ollama-based language models,
especially in the context of Retrieval-Augmented Generation (RAG) workflows.

It encapsulates reusable prompt formatting logic inside an enumeration class (`PromptOllama`),
which allows different types of structured prompts to be composed dynamically based on the use case.

Classes:
    PromptOllama (Enum-like): A class encapsulating prompt types for LLM interactions.
        - RAG_QA: Template for answering user questions using retrieved context from a knowledge base.

Example usage:
    from prompt_templates import PromptOllama

    prompt = PromptOllama.RAG_QA.prompt(
        query="What is reinforcement learning?",
        retrieval_context="Reinforcement learning is a machine learning technique that trains agents to make decisions by rewarding them for good actions."
    )
    
    # Send this prompt to the LLM for response
"""


class PromptOllama:
    """
    Enumeration of prompt templates for the Ollama chatbot.

    Currently defines a single template 'RAG_QA' for Retrieval-Augmented Generation (RAG).
    """

    RAG_QA = 1

    def prompt(self, query: str, retrieval_context: str) -> str:
        """
        Build a complete prompt string for the LLM.

        Args:
            query (str): The user's question.
            retrieval_context (str): The RAG-derived context to help with the answer.

        Returns:
            str: A formatted prompt string for the LLM.
        """
        system_instruction = (
            "You are a knowledgeable and helpful AI assistant. Use the provided context to help answer the user's question. "
            "If the context is incomplete, make your best effort to provide a coherent and helpful response based on your general knowledge. "
            "Never say 'I don't know'."
        )

        prompt_parts = [
            f"SYSTEM: {system_instruction}",
            f"CONTEXT: {retrieval_context}",
            f"USER QUESTION: {query}",
            "ASSISTANT:"
        ]

        return "\n\n".join(prompt_parts)
