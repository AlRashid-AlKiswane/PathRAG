"""
routes/__init__.py

Aggregates and exposes API route modules for the application.

Imported Routes:
- upload_route: Handles file uploads.
- chunking_route: Handles document chunking operations.
- embedding_chunks_route: Handles embedding operations on chunks.
- live_retrieval_route: PathRAG live semantic retrieval endpoint.
- storage_management_route: Storage management endpoints for database tables.
- chatbot_route: Chatbot conversational interface routes.
- resource_monitor_router: System resource monitoring endpoints.
- build_pathrag_route: Endpoint to build the PathRAG semantic graph.

This module centralizes route imports to simplify app router registration.
"""

from .upload_files import upload_route
from .chunking_docs import chunking_router
from .embedding_chunks import embedding_chunks_route
from .live_retrevel import live_retrieval_route
from .storage_management import storage_management_route
from .chatbot import chatbot_route
from .resource_monitor import resource_monitor_router
from .build_path_rag import build_pathrag_route
from .route_chunker_md_files import md_chunker_routes
from .user_file import user_file_route
__all__ = [
    "upload_route",
    "chunking_router",
    "embedding_chunks_route",
    "live_retrieval_route",
    "storage_management_route",
    "chatbot_route",
    "resource_monitor_router",
    "build_pathrag_route",
    "md_chunker_routes",
    "user_file_route"
]
