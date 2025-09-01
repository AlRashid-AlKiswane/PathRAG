"""
chatbot_route.py

Enhanced module implementing FastAPI endpoint for chatbot with dual-graph RAG system.
This version maintains the main graph for general information while supporting user-specific 
graphs for uploaded documents, ensuring comprehensive retrieval from both sources.

Features:
---------
- Dual retrieval system: User graph + Main graph
- Main graph preservation during user graph loading
- Combined scoring and ranking of paths from both graphs
- Intelligent fallback to main graph when user graph is insufficient
- Enhanced prompt generation with multi-source context
- Comprehensive logging and error handling

Route:
------
    POST /api/v1/chatbot

Author:
-------
ALRashid AlKiswane (Enhanced Version)
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pymongo import MongoClient

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from src.llms_providers import OllamaModel
from src.mongodb import insert_chatbot_entry_to_mongo
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import get_mongo_db, get_llm, get_path_rag

from src.rag import PathRAG
from src.prompt import PromptOllama
from src.schemas import Chatbot

# Initialize logger and settings
logger = setup_logging(name="CHATBOT-WORKFLOW")
app_settings: Settings = get_settings()

chatbot_route = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["Chatbot"],
    responses={404: {"description": "Not found"}}
)


class DualGraphRAG:
    """
    Handles retrieval from both main graph (general knowledge) and user graph (uploaded docs)
    """
    
    def __init__(self, pathrag: PathRAG):
        self.pathrag = pathrag
        self.main_graph_path = None
        self.original_graph_state = None
        
    def backup_main_graph(self):
        """Store reference to main graph path for restoration"""
        try:
            # Store the current graph state info instead of deep copying
            if hasattr(self.pathrag, 'graph_path') and self.pathrag.graph_path:
                self.main_graph_path = self.pathrag.graph_path
                logger.debug("Main graph path stored: %s", self.main_graph_path)
            elif hasattr(self.pathrag, 'graph') and self.pathrag.graph:
                # If no path available, store essential graph attributes
                self.original_graph_state = {
                    'has_graph': True,
                    'graph_type': type(self.pathrag.graph).__name__
                }
                logger.debug("Main graph state information stored")
            else:
                logger.warning("No main graph found to backup")
        except Exception as e:
            logger.error("Failed to backup main graph info: %s", e)
            
    def restore_main_graph(self):
        """Restore the main graph from stored path or reinitialize"""
        try:
            if self.main_graph_path and os.path.exists(self.main_graph_path):
                # Reload from the original main graph file
                self.pathrag.load_graph(file_path=self.main_graph_path)
                logger.debug("Main graph restored from path: %s", self.main_graph_path)
            elif self.original_graph_state and self.original_graph_state.get('has_graph'):
                # If we had a graph but no path, we need to reinitialize
                # This assumes there's a default main graph loading mechanism
                logger.warning("Main graph had no file path, attempting to reinitialize...")
                # You might need to add a method to reload the default main graph
                # self.pathrag.load_default_graph()  # Implement this method if needed
            else:
                logger.warning("No main graph backup available to restore")
        except Exception as e:
            logger.error("Failed to restore main graph: %s", e)
            
class DualGraphRAG:
    """
    Handles retrieval from both main graph (general knowledge) and user graph (uploaded docs)
    Uses separate instances to avoid conflicts
    """
    
    def __init__(self, main_pathrag: PathRAG):
        self.main_pathrag = main_pathrag
        self.user_pathrag = None
        
    def create_user_pathrag_instance(self, user_id: str) -> bool:
        """Create a separate PathRAG instance for user graph"""
        try:
            # Check if user has uploaded documents
            GRAPH_DIR = os.path.join(MAIN_DIR, "pathrag_data", user_id)
            user_graph_path = os.path.join(GRAPH_DIR, f"{user_id}.pkl")
            
            if os.path.exists(user_graph_path):
                logger.info("Creating user PathRAG instance for user_id='%s'", user_id)
                
                # Create a new PathRAG instance for user graph
                # This assumes PathRAG can be instantiated without parameters
                # or you can pass the same initialization parameters as the main one
                try:
                    self.user_pathrag = PathRAG()  # You might need to pass initialization params
                    self.user_pathrag.load_graph(file_path=user_graph_path)
                    logger.debug("User PathRAG instance created and loaded successfully")
                    return True
                except Exception as e:
                    logger.error("Failed to create user PathRAG instance: %s", e)
                    self.user_pathrag = None
                    return False
            else:
                logger.info("No user graph found for user_id='%s'", user_id)
                return False
                
        except Exception as e:
            logger.error("Error creating user PathRAG instance: %s", e)
            return False
        
    def retrieve_from_user_graph(self, user_id: str, query: str, top_k: int, max_hop: int) -> Tuple[List, List, str]:
        """
        Retrieve from user-specific graph using separate instance
        
        Returns:
            Tuple[nodes, paths, retrieval_context]: Results from user graph
        """
        user_nodes, user_paths, user_context = [], [], ""
        
        try:
            # Create user PathRAG instance if user has documents
            if self.create_user_pathrag_instance(user_id):
                # Perform retrieval from user graph
                user_nodes = self.user_pathrag.retrieve_nodes(query=query, top_k=top_k)
                user_paths = self.user_pathrag.prune_paths(nodes=user_nodes, max_hops=max_hop)
                
                if user_paths:
                    scored_user_paths = self.user_pathrag.score_paths(user_paths)
                    user_context = self.user_pathrag.generate_prompt(query=query, scored_paths=scored_user_paths)
                    logger.info("Retrieved %d paths from user graph", len(scored_user_paths))
                else:
                    logger.info("No valid paths found in user graph")
                    
        except Exception as e:
            logger.error("Error retrieving from user graph: %s", e)
            
        return user_nodes, user_paths, user_context
        
    def retrieve_from_main_graph(self, query: str, top_k: int, max_hop: int) -> Tuple[List, List, str]:
        """
        Retrieve from main graph (general knowledge) using main instance
        
        Returns:
            Tuple[nodes, paths, retrieval_context]: Results from main graph
        """
        main_nodes, main_paths, main_context = [], [], ""
        
        try:
            # Use the main PathRAG instance (never modified)
            main_nodes = self.main_pathrag.retrieve_nodes(query=query, top_k=top_k)
            main_paths = self.main_pathrag.prune_paths(nodes=main_nodes, max_hops=max_hop)
            
            if main_paths:
                scored_main_paths = self.main_pathrag.score_paths(main_paths)
                main_context = self.main_pathrag.generate_prompt(query=query, scored_paths=scored_main_paths)
                logger.info("Retrieved %d paths from main graph", len(scored_main_paths))
            else:
                logger.info("No valid paths found in main graph")
                
        except Exception as e:
            logger.error("Error retrieving from main graph: %s", e)
            
        return main_nodes, main_paths, main_context
        
    def cleanup_user_pathrag(self):
        """Clean up user PathRAG instance to free resources"""
        try:
            if self.user_pathrag:
                # Clean up user PathRAG instance
                self.user_pathrag = None
                logger.debug("User PathRAG instance cleaned up")
        except Exception as e:
            logger.error("Error cleaning up user PathRAG: %s", e)
        
    def combine_retrieval_contexts(self, user_context: str, main_context: str, query: str) -> str:
        """
        Build a structured prompt by combining user and main graph contexts,
        or fall back to a professional default if no context is found.
        """

        # Base fallback if no context is available
        prompt = f"""
        You are a knowledgeable and professional AI assistant. A user has asked the following question:

        Question:
        "{query}"

        If no relevant information is available in the retrieved knowledge, please:
        1. Acknowledge that the exact answer is not found in the available information
        2. Suggest what type of information or resources might help answer their question
        3. Encourage the user to rephrase or provide more context
        4. Maintain a supportive and professional tone
        """.strip()

        # Add contextual knowledge if available
        context_sections = []
        if user_context:
            context_sections.append(f"USER DOCUMENTS CONTEXT:\n{user_context}")
            logger.info("Using user graph context")
        if main_context:
            context_sections.append(f"GENERAL KNOWLEDGE CONTEXT:\n{main_context}")
            logger.info("Using main graph context")
        if context_sections:
            prompt += "\n\nThe following information was retrieved and may be relevant:\n\n"
            prompt += "\n\n---\n\n".join(context_sections)
            logger.info("Final prompt includes contextual knowledge")
        else:
            logger.warning("No context found, using fallback only")
        prompt += "\n\nNow, please provide a clear, accurate, and professional response based on the available information."
        return prompt


@chatbot_route.post("", response_class=JSONResponse)
async def chatbot(
    body: Chatbot,
    db: MongoClient = Depends(get_mongo_db),
    llm: OllamaModel = Depends(get_llm),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Enhanced chatbot handler with dual-graph RAG system.
    
    Performs retrieval from both user-specific graphs (uploaded documents) and 
    the main graph (general knowledge) to provide comprehensive answers.

    Args:
        body (Chatbot): Input payload including query, top_k, temperature, max_tokens, user_id, and caching options.
        db (MongoClient): MongoDB client, injected by dependency.
        llm (OllamaModel): Injected local LLM instance (Ollama).
        pathrag (PathRAG): Path-aware RAG engine from app state.

    Returns:
        JSONResponse: Generated answer and cache status with dual-source retrieval.
    """
    try:
        # Unpack request body
        query = body.query
        top_k = body.top_k
        max_hop = body.max_hop or 2  # default hops
        temperature = body.temperature
        max_new_tokens = body.max_new_tokens
        max_input_tokens = body.max_input_tokens
        user_id = body.user_id
        cache = body.cache

        logger.info("Enhanced chatbot request | user_id='%s' | query='%s' | cache=%s", user_id, query, cache)

        # Step 1: Check Cache
        if cache:
            cached_entry = await check_cache(db, user_id, query)
            if cached_entry:
                logger.info("Cache hit for user_id='%s' and query='%s'", user_id, query)
                return JSONResponse(content={"response": cached_entry["llm_response"], "cached": True})
            else:
                logger.debug("Cache miss for user_id='%s' and query='%s'", user_id, query)

        # Step 2: Initialize Dual Graph RAG
        dual_rag = DualGraphRAG(pathrag)
        
        # Step 3: Retrieve from User Graph (if exists)
        logger.debug("Performing retrieval from user graph...")
        user_nodes, user_paths, user_context = dual_rag.retrieve_from_user_graph(
            user_id=user_id, 
            query=query, 
            top_k=top_k, 
            max_hop=max_hop
        )
        
        # Step 4: Retrieve from Main Graph
        logger.debug("Performing retrieval from main graph...")
        main_nodes, main_paths, main_context = dual_rag.retrieve_from_main_graph(
            query=query, 
            top_k=top_k, 
            max_hop=max_hop
        )
        
        # Step 5: Combine Retrieval Contexts
        combined_context = dual_rag.combine_retrieval_contexts(
            user_context=user_context,
            main_context=main_context,
            query=query
        )
        
        if not combined_context.strip():
            logger.warning("No valid context found from either graph for query: %s", query)
            return JSONResponse(
                status_code=200,
                content={
                    "message": "[!] No relevant information found in either user documents or general knowledge base. Try rephrasing your query.",
                    "cached": False
                }
            )

        # Step 6: Generate Enhanced Prompt
        logger.debug("Generating enhanced prompt with dual-source context.")
        prompt_template = PromptOllama()
        
        # Enhanced prompt with source indication
        enhanced_prompt = prompt_template.prompt(
            query=query, 
            retrieval_context=combined_context
        )

        # Step 7: Generate LLM Response
        logger.debug("Calling Ollama LLM to generate response with dual-source context.")
        llm_response = llm.generate(
            prompt=enhanced_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens
        )

        if not llm_response or llm_response.startswith("[ERROR]"):
            logger.error("LLM failed to generate a valid response.")
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid response.")

        logger.info("Enhanced LLM response generated successfully with dual-source context.")

        # Step 8: Store in Cache/Log with Enhanced Context
        try:
            success = insert_chatbot_entry_to_mongo(
                db=db,
                user_id=user_id,
                query=query,
                llm_response=llm_response,
                retrieval_context=combined_context,
                retrieval_rank=0,
                doc_id=None,
            )
            if not success:
                logger.warning("Failed to insert enhanced chatbot entry for user_id='%s'", user_id)
            else:
                logger.debug("💾 Enhanced chatbot entry committed to database.")
        except Exception as db_err:
            logger.exception("Failed to store enhanced chatbot entry in DB: %s", db_err)

        # Step 9: Prepare Response with Source Information
        response_data = {
            "response": llm_response,
            "cached": False,
            "sources": {
                "user_documents": bool(user_context.strip()),
                "general_knowledge": bool(main_context.strip()),
                "user_paths_count": len(user_paths) if user_paths else 0,
                "main_paths_count": len(main_paths) if main_paths else 0
            }
        }
        
        # Step 10: Cleanup resources
        dual_rag.cleanup_user_pathrag()

        return JSONResponse(content=response_data)

    except HTTPException as http_err:
        logger.warning("HTTPException in enhanced chatbot: %s", http_err.detail)
        raise

    except Exception as e:
        logger.exception("Unexpected error in enhanced chatbot route: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during enhanced chatbot operation."
        )


async def check_cache(db: MongoClient, user_id: str, query: str) -> Optional[dict]:
    """
    Checks the database for a cached chatbot response matching user_id and query.

    Args:
        db (MongoClient): MongoDB client.
        user_id (str): User identifier.
        query (str): User query text.

    Returns:
        Optional[dict]: Cached entry document or None if not found.
    """
    try:
        cached = db.chatbot.find_one({"user_id": user_id, "query": query})
        return cached
    except Exception as e:
        logger.error("Error checking cache for user_id='%s' query='%s': %s", user_id, query, e)
        return None


# Optional: Health check endpoint for dual-graph system
@chatbot_route.get("/health")
async def health_check(pathrag: PathRAG = Depends(get_path_rag)):
    """
    Health check endpoint for the enhanced dual-graph chatbot system.
    """
    try:
        # Basic health check
        return JSONResponse(content={
            "status": "healthy",
            "system": "dual-graph RAG chatbot",
            "main_graph_loaded": hasattr(pathrag, 'graph') and pathrag.graph is not None,
            "timestamp": logger.name
        })
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )
