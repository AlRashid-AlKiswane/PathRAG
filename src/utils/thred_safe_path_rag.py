"""
threadsafe_rag.py

This module provides thread-safe and concurrent management utilities for the PathRAG
(Path-aware Retrieval-Augmented Generation) system. It ensures safe multi-threaded
access to graph operations and LLM/embedding models, allowing concurrent embedding
generation, model inference, and graph manipulation without race conditions.

Main Components:
----------------
1. ThreadSafePathRAG:
   - A thread-safe wrapper around a PathRAG instance.
   - Provides locks for safe read, write, and graph operations.
   - Allows multiple concurrent readers and exclusive write access.

2. ConcurrentModelManager:
   - Manages LLM and embedding model instances for concurrent access.
   - Uses thread-local storage to provide each thread its own model instance.
   - Supports asynchronous generation and embedding using thread pools.
   - Allows safe shutdown of resources when no longer needed.

3. run_in_threadpool:
   - A decorator to convert synchronous functions to asynchronous by executing
     them in a thread pool.

Usage:
------
- Wrap your PathRAG instance with ThreadSafePathRAG to ensure thread safety.
- Use ConcurrentModelManager to manage multiple threads accessing LLMs or embedding
  models.
- Use the `run_in_threadpool` decorator for synchronous functions that need to run
  asynchronously.

Example:
--------
    safe_rag = ThreadSafePathRAG(path_rag_instance)
    manager = ConcurrentModelManager(MyLLM, MyEmbed, llm_config, embed_config, max_workers=4)

Dependencies:
-------------
- asyncio
- threading
- concurrent.futures
- Python 3.8+
- PathRAG system and related models

Author:
-------
ALRashid AlKiswane
"""
import os
import sys
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Lock, RLock
import threading

# === Configure Project Path ===
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except Exception as e:
    logging.critical("Failed to configure project path: %s", e, exc_info=True)
    sys.exit(1)

# === Internal Imports ===
try:
    from src.infra import setup_logging
    from src.helpers import get_settings, Settings
except ImportError as e:
    logging.critical("Import error during module loading: %s", e, exc_info=True)
    sys.exit(1)

# === Initialize Logging and Settings ===
logger = setup_logging(name="MAIN")
config: Settings = get_settings()

# === Thread Safety Components ===
class ThreadSafePathRAG:
    """
    A thread-safe wrapper for PathRAG operations, ensuring that graph operations
    are performed safely in a multi-threaded environment.

    Attributes:
        _path_rag (PathRAG): The PathRAG instance to be wrapped.
        _read_lock (RLock): A reentrant lock for read operations.
        _write_lock (Lock): A non-reentrant lock for write operations.
        _graph_lock (RLock): A reentrant lock for graph-specific operations.
    """

    def __init__(self, path_rag_instance):
        """
        Initializes the ThreadSafePathRAG wrapper with the given PathRAG instance.

        Args:
            path_rag_instance (PathRAG): The PathRAG instance to be wrapped.
        """
        self._path_rag = path_rag_instance
        self._read_lock = RLock()  # Allows multiple readers
        self._write_lock = Lock()   # Exclusive write access
        self._graph_lock = RLock()  # For graph operations

    def get_graph(self):
        """
        Retrieves the graph in a thread-safe manner.

        Returns:
            Graph: The underlying graph object.
        """
        with self._read_lock:
            return self._path_rag.g
    
    def load_graph(self, path):
        """
        Loads a graph from the specified path in a thread-safe manner.

        Args:
            path (str): The file path to load the graph from.

        Returns:
            bool: True if the graph was loaded successfully, False otherwise.
        """
        with self._write_lock:
            return self._path_rag.load_graph(path)
    
    def save_graph(self, path: str = "./pathrag_data/checkpoints/graph.pkl"):
        """
        Saves the current graph to the specified path in a thread-safe manner.

        Args:
            path (str): The file path to save the graph to.

        Returns:
            bool: True if the graph was saved successfully, False otherwise.
        """
        with self._write_lock:
            return self._path_rag.save_graph(path)
    
    def add_to_graph(self, *args, **kwargs):
        """
        Adds data to the graph in a thread-safe manner.

        Args:
            *args: Positional arguments to be passed to the add method.
            **kwargs: Keyword arguments to be passed to the add method.

        Returns:
            bool: True if the data was added successfully, False otherwise.
        """
        with self._write_lock:
            return self._path_rag.add_to_graph(*args, **kwargs)
    
    def retrieve(self, *args, **kwargs):
        """
        Retrieves data from the graph in a thread-safe manner.

        Args:
            *args: Positional arguments to be passed to the retrieve method.
            **kwargs: Keyword arguments to be passed to the retrieve method.

        Returns:
            Any: The data retrieved from the graph.
        """
        with self._read_lock:
            return self._path_rag.retrieve(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegates attribute access to the underlying PathRAG instance.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Any: The attribute value from the underlying PathRAG instance.
        """
        return getattr(self._path_rag, name)

# === Async Utilities ===
def run_in_threadpool(func):
    """
    A decorator to run synchronous functions in a thread pool asynchronously.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: An asynchronous wrapper for the function.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        """
        Executes the wrapped function in a thread pool.

        Args:
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            Any: The result of the function execution.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

class ConcurrentModelManager:
    """
    Manages model instances for concurrent access, ensuring that each thread
    has its own instance of the models to prevent race conditions.

    Attributes:
        llm_model_class (type): The class of the LLM model.
        embedding_model_class (type): The class of the embedding model.
        llm_config (Settings): Configuration settings for the LLM model.
        embedding_config (Settings): Configuration settings for the embedding model.
        _llm_pool (ThreadPoolExecutor): Executor for LLM model operations.
        _embedding_pool (ThreadPoolExecutor): Executor for embedding model operations.
        _llm_instances (threading.local): Thread-local storage for LLM instances.
        _embedding_instances (threading.local): Thread-local storage for embedding instances.
    """

    def __init__(self, llm_model_class, embedding_model_class, llm_config, embedding_config, max_workers: int = 4):
        """
        Initializes the ConcurrentModelManager with the given model classes and configurations.

        Args:
            llm_model_class (type): The class of the LLM model.
            embedding_model_class (type): The class of the embedding model.
            llm_config (Settings): Configuration settings for the LLM model.
            embedding_config (Settings): Configuration settings for the embedding model.
            max_workers (int): The maximum number of workers for the thread pools.
        """
        self.llm_model_class = llm_model_class
        self.embedding_model_class = embedding_model_class
        self.llm_config = llm_config
        self.embedding_config = embedding_config
        self._llm_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm_worker")
        self._embedding_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="embed_worker")
        self._llm_instances = threading.local()
        self._embedding_instances = threading.local()

    def get_llm_instance(self):
        """
        Retrieves the thread-local LLM model instance.

        Returns:
            LLMModel: The LLM model instance.
        """
        if not hasattr(self._llm_instances, 'model'):
            self._llm_instances.model = self.llm_model_class(self.llm_config)
        return self._llm_instances.model
    
    def get_embedding_instance(self):
        """
        Retrieves the thread-local embedding model instance.

        Returns:
            EmbeddingModel: The embedding model instance.
        """
        if not hasattr(self._embedding_instances, 'model'):
            self._embedding_instances.model = self.embedding_model_class(self.embedding_config)
        return self._embedding_instances.model
    
    async def generate_async(self, *args, **kwargs):
        """
        Asynchronously generates output using the LLM model.

        Args:
            *args: Positional arguments to be passed to the generate method.
            **kwargs: Keyword arguments to be passed to the generate method.

        Returns:
            Any: The generated output from the LLM model.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._llm_pool, 
            lambda: self.get_llm_instance().generate(*args, **kwargs)
        )
    
    async def embed_async(self, *args, **kwargs):
        """
        Asynchronously generates embeddings using the embedding model.

        Args:
            *args: Positional arguments to be passed to the embed method.
            **kwargs: Keyword arguments to be passed to the embed method.

        Returns:
            Any: The generated embeddings from the embedding model.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._embedding_pool,
            lambda: self.get_embedding_instance().embed(*args, **kwargs)
        )
    
    def shutdown(self):
        """
        Shuts down the thread pools, releasing any resources.

        This method should be called when the application is shutting down
        to ensure that all resources are properly released.
        """
        self._llm_pool.shutdown(wait=True)
        self._embedding_pool.shutdown(wait=True)
