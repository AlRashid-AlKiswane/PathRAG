"""
chatbot_route.py

This module implements the FastAPI endpoint for interacting with a chatbot powered by
Retrieval-Augmented Generation (RAG). It leverages a local Ollama LLM along with a 
semantic graph engine (PathRAG) to generate high-quality, context-aware answers to user queries.

The endpoint accepts structured input (query, user ID, LLM parameters, etc.), optionally 
checks for a cached response in MongoDB, performs semantic retrieval from a knowledge graph,
constructs a prompt, and invokes a local language model to generate a relevant reply.

Features:
---------
- Retrieval-augmented generation using PathRAG graph reasoning.
- Prompt creation based on scored and pruned semantic paths.
- Integration with a local Ollama LLM model.
- Optional response caching in MongoDB.
- Comprehensive logging and error handling.

Route:
------
    POST /api/v1/chatbot

Input Schema (Chatbot):
-----------------------
- query: str - The user’s natural language question.
- user_id: str - A unique identifier for the user (used for caching).
- top_k: int - Number of top relevant chunks to retrieve.
- max_hop: Optional[int] - Maximum hops allowed in graph traversal.
- temperature: float - LLM response sampling temperature.
- max_new_tokens: int - Max number of new tokens to generate.
- max_input_tokens: int - Token limit for input prompt.
- cache: bool - Whether to enable response caching.

Output:
-------
JSON response containing:
- "response": Generated text response from the LLM.
- "cached": Boolean indicating if response came from cache.

Dependencies:
-------------
- FastAPI
- MongoDB (via PyMongo or Motor)
- src.rag.PathRAG for graph-based retrieval
- src.llms_providers.OllamaModel for local inference
- src.prompt.PromptOllama for dynamic prompt formatting
- src.graph_db for MongoDB storage
- src.helpers and src.infra for configuration and logging

Author:
-------
ALRashid AlKiswane
"""

import os
import sys
import logging
from typing import Optional
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
from src.graph_db import insert_chatbot_entry_to_mongo
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


@chatbot_route.post("", response_class=JSONResponse)
async def chatbot(
    body: Chatbot,
    db: MongoClient = Depends(get_mongo_db),
    llm: OllamaModel = Depends(get_llm),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Handles chatbot requests using retrieval-augmented generation (RAG) and a local LLM.

    Args:
        body (Chatbot): Input payload including query, top_k, temperature, max_tokens, user_id, and caching options.
        db (MongoClient): MongoDB client, injected by dependency.
        llm (OllamaModel): Injected local LLM instance (Ollama).
        pathrag (PathRAG): Path-aware RAG engine from app state.

    Returns:
        JSONResponse: Generated answer and cache status.
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

        logger.info("Chatbot request | user_id='%s' | query='%s' | cache=%s", user_id, query, cache)

        # Step 1: Check Cache
        if cache:
            cached_entry = await check_cache(db, user_id, query)
            if cached_entry:
                logger.info("Cache hit for user_id='%s' and query='%s'", user_id, query)
                return JSONResponse(content={"response": cached_entry["llm_response"], "cached": True})
            else:
                logger.debug("Cache miss for user_id='%s' and query='%s'", user_id, query)

        # Step 2: Semantic Retrieval
        logger.debug("Performing semantic retrieval using PathRAG.")
        nodes = pathrag.retrieve_nodes(query=query, top_k=top_k)
        paths = pathrag.prune_paths(nodes=nodes, max_hops=max_hop)

        if not paths:
            logger.warning("No valid paths found for query: %s", query)
            return JSONResponse(
                status_code=200,
                content={"message": "[!] No valid paths found. Try lowering prune_thresh or increasing max_hops."}
            )

        scored_paths = pathrag.score_paths(paths)
        final_retrieval = pathrag.generate_prompt(query=query, scored_paths=scored_paths)
        logger.info("Retrieved and scored %d semantic paths.", len(scored_paths))

        # Step 3: Prompt Generation
        logger.debug("Generating prompt from top paths.")
        prompt_template = PromptOllama()
        prompt = prompt_template.prompt(query=query, retrieval_context=final_retrieval)

        # Step 4: Generate LLM Response
        logger.debug("Calling Ollama LLM to generate response.")
        llm_response = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens
        )

        if not llm_response or llm_response.startswith("[ERROR]"):
            logger.error("LLM failed to generate a valid response.")
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid response.")

        logger.info("LLM response generated successfully.")

        # Step 5: Store in Cache/Log
        try:
            success = insert_chatbot_entry_to_mongo(
                db=db,
                user_id=user_id,
                query=query,
                llm_response=llm_response,
                retrieval_context=final_retrieval,
                retrieval_rank=0,  # single combined retrieval rank
                doc_id=None,       # No specific doc id for combined retrieval
            )
            if not success:
                logger.warning("Failed to insert chatbot entry for user_id='%s'", user_id)
            else:
                logger.debug("💾 Chatbot entry committed to database.")
        except Exception as db_err:
            logger.exception("Failed to store chatbot entry in DB: %s", db_err)

        return JSONResponse(content={"response": llm_response, "cached": False})

    except HTTPException as http_err:
        logger.warning("HTTPException: %s", http_err.detail)
        raise

    except Exception as e:
        logger.exception("Unexpected error in chatbot route: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chatbot operation."
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
