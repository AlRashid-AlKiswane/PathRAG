"""
utils package

This package provides utility functions and helper classes used across the project,
including model management and file size handling.

Modules:
    - ollama_manager: Contains the `OllamaManager` class for managing interactions with the Ollama LLM backend.
    - size_file: Provides the `get_size` function to calculate the size of files in a human-readable format.

Example usage:
    from utils import OllamaManager, get_size

    manager = OllamaManager()
    file_size = get_size("/path/to/file.txt")
"""

from .ollama_maneger import OllamaManager
from .size_file import get_size
from .do_senitize import sanitize
