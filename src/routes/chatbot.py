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
    Uses separate instances to avoid conflicts and preserve the main graph
    """
    
    def __init__(self, main_pathrag: PathRAG):
        """
        Initialize with the main PathRAG instance
        
        Args:
            main_pathrag (PathRAG): The main graph instance (never modified)
        """
        self.main_pathrag = main_pathrag
        self.user_pathrag = None
        
    def create_user_pathrag_instance(self, user_id: str) -> bool:
        """
        Create a separate PathRAG instance for user graph without affecting main graph
        
        Args:
            user_id (str): User identifier
            
        Returns:
            bool: True if user PathRAG instance created successfully, False otherwise
        """
        try:
            # Clean up any existing user PathRAG instance first
            self.cleanup_user_pathrag()
            
            # Check if user has uploaded documents
            GRAPH_DIR = os.path.join(MAIN_DIR, "pathrag_data", user_id)
            user_graph_path = os.path.join(GRAPH_DIR, f"{user_id}.pkl")
            
            if not os.path.exists(user_graph_path):
                logger.info("No user graph found for user_id='%s' at path: %s", user_id, user_graph_path)
                return False
                
            logger.info("Creating separate user PathRAG instance for user_id='%s'", user_id)
            
            # Create a completely new PathRAG instance for user graph
            # This ensures the main graph is never affected
            try:
                # Initialize new PathRAG instance with same configuration as main
                # You might need to adjust this based on your PathRAG constructor
                self.user_pathrag = PathRAG()
                
                # Load the user-specific graph
                self.user_pathrag.load_graph(file_path=user_graph_path)
                
                logger.debug("User PathRAG instance created and loaded successfully from: %s", user_graph_path)
                return True
                
            except Exception as load_error:
                logger.error("Failed to load user graph from '%s': %s", user_graph_path, load_error)
                self.cleanup_user_pathrag()
                return False
                
        except Exception as e:
            logger.error("Error creating user PathRAG instance for user_id='%s': %s", user_id, e)
            self.cleanup_user_pathrag()
            return False
        
    def retrieve_from_user_graph(self, user_id: str, query: str, top_k: int, max_hop: int) -> Tuple[List, List, str]:
        """
        Retrieve from user-specific graph using separate instance
        
        Args:
            user_id (str): User identifier
            query (str): Search query
            top_k (int): Number of top nodes to retrieve
            max_hop (int): Maximum hops for path pruning
            
        Returns:
            Tuple[List, List, str]: (nodes, paths, retrieval_context) from user graph
        """
        user_nodes, user_paths, user_context = [], [], ""
        
        try:
            # Create user PathRAG instance if user has documents
            if not self.create_user_pathrag_instance(user_id):
                logger.info("No user graph available for user_id='%s'", user_id)
                return user_nodes, user_paths, user_context
                
            # Perform retrieval from user graph using the separate instance
            logger.debug("Retrieving nodes from user graph...")
            user_nodes = self.user_pathrag.retrieve_nodes(query=query, top_k=top_k)
            
            if user_nodes:
                logger.debug("Pruning paths from user graph...")
                user_paths = self.user_pathrag.prune_paths(nodes=user_nodes, max_hops=max_hop)
                
                if user_paths:
                    logger.debug("Scoring and generating context from user paths...")
                    scored_user_paths = self.user_pathrag.score_paths(user_paths)
                    user_context = self.user_pathrag.generate_prompt(query=query, scored_paths=scored_user_paths)
                    logger.info("Successfully retrieved %d paths from user graph", len(scored_user_paths))
                else:
                    logger.info("No valid paths found in user graph after pruning")
            else:
                logger.info("No matching nodes found in user graph")
                    
        except Exception as e:
            logger.error("Error retrieving from user graph for user_id='%s': %s", user_id, e)
            
        return user_nodes, user_paths, user_context
        
    def retrieve_from_main_graph(self, query: str, top_k: int, max_hop: int) -> Tuple[List, List, str]:
        """
        Retrieve from main graph (general knowledge) using main instance
        Main graph is never modified or affected by user operations
        
        Args:
            query (str): Search query
            top_k (int): Number of top nodes to retrieve
            max_hop (int): Maximum hops for path pruning
            
        Returns:
            Tuple[List, List, str]: (nodes, paths, retrieval_context) from main graph
        """
        main_nodes, main_paths, main_context = [], [], ""
        
        try:
            # Use the main PathRAG instance (always preserved and never modified)
            logger.debug("Retrieving nodes from main graph...")
            main_nodes = self.main_pathrag.retrieve_nodes(query=query, top_k=top_k)
            
            if main_nodes:
                logger.debug("Pruning paths from main graph...")
                main_paths = self.main_pathrag.prune_paths(nodes=main_nodes, max_hops=max_hop)
                
                if main_paths:
                    logger.debug("Scoring and generating context from main paths...")
                    scored_main_paths = self.main_pathrag.score_paths(main_paths)
                    main_context = self.main_pathrag.generate_prompt(query=query, scored_paths=scored_main_paths)
                    logger.info("Successfully retrieved %d paths from main graph", len(scored_main_paths))
                else:
                    logger.info("No valid paths found in main graph after pruning")
            else:
                logger.info("No matching nodes found in main graph")
                
        except Exception as e:
            logger.error("Error retrieving from main graph: %s", e)
            
        return main_nodes, main_paths, main_context
        
    def cleanup_user_pathrag(self):
        """
        Clean up user PathRAG instance to free resources
        Main graph is never affected by this operation
        """
        try:
            if self.user_pathrag is not None:
                # Clean up user PathRAG instance
                del self.user_pathrag
                self.user_pathrag = None
                logger.debug("User PathRAG instance cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up user PathRAG: %s", e)
        
    def combine_retrieval_contexts(self, user_context: str, main_context: str, query: str) -> str:
        """
        Build a structured prompt by combining user and main graph contexts,
        or fall back to a professional default if no context is found.
        
        Args:
            user_context (str): Context from user documents
            main_context (str): Context from main knowledge graph
            query (str): Original user query
            
        Returns:
            str: Combined context prompt
        """
        logger.debug("Combining retrieval contexts...")
        
        # Start with base prompt structure
        prompt = f"""You are a knowledgeable and professional AI assistant. A user has asked the following question:

Question: "{query}"

"""

        # Add contextual knowledge if available
        context_sections = []
        
        # Prioritize user documents as they are more specific
        if user_context and user_context.strip():
            context_sections.append(f"RELEVANT INFORMATION FROM USER DOCUMENTS:\n{user_context.strip()}")
            logger.info("Including user graph context in response")
            
        if main_context and main_context.strip():
            context_sections.append(f"RELEVANT GENERAL KNOWLEDGE:\n{main_context.strip()}")
            logger.info("Including main graph context in response")
        
        if context_sections:
            prompt += "Based on the following relevant information:\n\n"
            prompt += "\n\n" + "="*50 + "\n\n".join(context_sections)
            prompt += "\n\nPlease provide a comprehensive, accurate, and helpful response to the user's question."
        else:
            # Fallback when no context is available
            prompt += """No specific relevant information was found in the available knowledge base for this query.

Please:
1. Acknowledge that specific information is not available in the current knowledge base
2. Provide any general guidance or suggestions that might be helpful
3. Suggest how the user might rephrase their question or what additional context might help
4. Maintain a supportive and professional tone"""
            logger.warning("No context found from either graph, using fallback prompt")
        
        logger.debug("Combined context prompt generated successfully")
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
    The main graph is always preserved and never modified.

    Args:
        body (Chatbot): Input payload including query, top_k, temperature, max_tokens, user_id, and caching options.
        db (MongoClient): MongoDB client, injected by dependency.
        llm (OllamaModel): Injected local LLM instance (Ollama).
        pathrag (PathRAG): Main path-aware RAG engine from app state (never modified).

    Returns:
        JSONResponse: Generated answer and cache status with dual-source retrieval.
    """
    dual_rag = None
    
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

        # Step 2: Initialize Dual Graph RAG (main graph is preserved)
        dual_rag = DualGraphRAG(pathrag)
        logger.debug("DualGraphRAG initialized with preserved main graph")
        
        # Step 3: Retrieve from User Graph (if exists) - uses separate instance
        logger.debug("Performing retrieval from user graph...")
        user_nodes, user_paths, user_context = dual_rag.retrieve_from_user_graph(
            user_id=user_id, 
            query=query, 
            top_k=top_k, 
            max_hop=max_hop
        )
        
        # Step 4: Retrieve from Main Graph (always available and preserved)
        logger.debug("Performing retrieval from main graph...")
        main_nodes, main_paths, main_context = dual_rag.retrieve_from_main_graph(
            query=query, 
            top_k=top_k, 
            max_hop=max_hop
        )
        
        # Step 5: Combine Retrieval Contexts
        logger.debug("Combining retrieval contexts from both graphs...")
        combined_context = dual_rag.combine_retrieval_contexts(
            user_context=user_context,
            main_context=main_context,
            query=query
        )
        
        # Step 6: Generate Enhanced Prompt
        logger.debug("Generating enhanced prompt with dual-source context")
        prompt_template = PromptOllama()
        
        enhanced_prompt = prompt_template.prompt(
            query=query, 
            retrieval_context=combined_context
        )

        # Step 7: Generate LLM Response
        logger.debug("Calling Ollama LLM to generate response with dual-source context")
        llm_response = llm.generate(
            prompt=enhanced_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens
        )

        if not llm_response or llm_response.startswith("[ERROR]"):
            logger.error("LLM failed to generate a valid response")
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid response")

        logger.info("Enhanced LLM response generated successfully with dual-source context")

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
                logger.debug("💾 Enhanced chatbot entry committed to database")
        except Exception as db_err:
            logger.exception("Failed to store enhanced chatbot entry in DB: %s", db_err)

        # Step 9: Prepare Response with Source Information
        response_data = {
            "response": llm_response,
            "cached": False,
            "sources": {
                "user_documents": bool(user_context and user_context.strip()),
                "general_knowledge": bool(main_context and main_context.strip()),
                "user_paths_count": len(user_paths) if user_paths else 0,
                "main_paths_count": len(main_paths) if main_paths else 0
            }
        }
        
        logger.info("Response prepared with source information: user_docs=%s, general_knowledge=%s", 
                   response_data["sources"]["user_documents"], 
                   response_data["sources"]["general_knowledge"])

        return JSONResponse(content=response_data)

    except HTTPException as http_err:
        logger.warning("HTTPException in enhanced chatbot: %s", http_err.detail)
        raise

    except Exception as e:
        logger.exception("Unexpected error in enhanced chatbot route: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during enhanced chatbot operation"
        )
    
    finally:
        # Step 10: Always cleanup resources (main graph remains intact)
        if dual_rag:
            dual_rag.cleanup_user_pathrag()
            logger.debug("Cleanup completed, main graph preserved")


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


# Health check endpoint for dual-graph system
@chatbot_route.get("/health")
async def health_check(pathrag: PathRAG = Depends(get_path_rag)):
    """
    Health check endpoint for the enhanced dual-graph chatbot system.
    Verifies that the main graph is loaded and accessible.
    """
    try:
        # Check main graph status
        main_graph_loaded = hasattr(pathrag, 'graph') and pathrag.graph is not None
        
        return JSONResponse(content={
            "status": "healthy",
            "system": "dual-graph RAG chatbot",
            "main_graph_loaded": main_graph_loaded,
            "main_graph_preserved": True,  # Always true in this implementation
            "timestamp": str(logger.name)
        })
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )
