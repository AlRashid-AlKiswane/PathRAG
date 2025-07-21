"""
Package for prompt-related utilities and classes.

This package provides prompt template implementations such as `PromptOllama` 
which can be used to generate prompts for language models or related tasks.
"""

from .prompt_templates import PromptOllama

__all__ = ["PromptOllama"]
