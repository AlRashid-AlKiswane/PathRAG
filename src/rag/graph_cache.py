

import logging
import os
import pickle
import sys
import threading
from typing import Any, Optional
import valkey

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.schemas import PathRAGConfig
from src.infra import setup_logging

logger = setup_logging(name="GRAPH-CACHE")
VALKEY_AVAILABLE = True

class GraphCache:
    """Intelligent caching system for graph operations."""
    
    def __init__(self, config: PathRAGConfig):
        self.config = config
        self._memory_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize Valkey client if caching is enabled
        self.valkey_client = None
        if VALKEY_AVAILABLE and config.enable_caching:
            try:
                self.valkey_client = valkey.Valkey(host='localhost', port=6379, db=0, decode_responses=False)
                self.valkey_client.ping()
                logger.info("Valkey cache initialized")
            except Exception as e:
                logger.warning(f"Valkey initialization failed: {e}")
                self.valkey_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Try memory cache first
        with self._cache_lock:
            if key in self._memory_cache:
                return self._memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set item in cache."""
        # Memory cache with size limit
        with self._cache_lock:
            if len(self._memory_cache) >= self.config.cache_size:
                # Remove oldest item (simple FIFO)
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
            self._memory_cache[key] = value
        
        # Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(key, expire, pickle.dumps(value))
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
