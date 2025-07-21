import os
import sys
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlite3 import Connection

# Set up project base directory
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.llms_providers import OllamaModel, HuggingFaceModel
from src.db import insert_chatbot_entry, pull_from_table
from src.infra import setup_logging
from src.helpers import get_settings, Settings
from src import (get_db_conn, get_llm, get_path_rag)

from src.rag import PathRAG
from src.prompt import PromptOllama
from src.schemas import Chatbot

# Initialize logger and settings
logger = setup_logging()
app_settings: Settings = get_settings()

chatbot_route = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["Chatbot"],
    responses={404: {"description": "Not found"}}
)


# === Endpoint ===
@chatbot_route.post("", response_class=JSONResponse)
async def chatbot(
    body: Chatbot,
    conn: Connection = Depends(get_db_conn),
    llm: OllamaModel = Depends(get_llm),
    pathrag: PathRAG = Depends(get_path_rag)
) -> JSONResponse:
    """
    Handles chatbot requests using retrieval-augmented generation (RAG) and a local LLM.

    Uses semantic graph traversal (PathRAG) to gather contextual information before generating
    an answer using the local Ollama model. Caches responses if enabled.

    Args:
        body (Chatbot): Input payload including query, top_k, temperature, max_tokens, user_id, and caching options.
        conn (Connection): SQLite connection dependency.
        llm (OllamaModel): Injected local LLM instance (Ollama).
        pathrag (PathRAG): Path-aware RAG engine from app state.

    Returns:
        JSONResponse: Generated answer and cache status.
    """
    try:
        # === Unpack request body ===
        query = body.query
        top_k = body.top_k
        max_hop = body.max_hop or 2  # Ensure default
        temperature = body.temperature
        max_new_tokens = body.max_new_tokens
        max_input_tokens = body.max_input_tokens
        user_id = body.user_id
        cache = body.cache

        logger.info("🚀 Chatbot request | user_id='%s' | query='%s' | cache=%s", user_id, query, cache)

        # === Step 1: Check Cache ===
        if cache:
            logger.debug("🔍 Checking cache for user_id='%s' and query='%s'", user_id, query)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT llm_response FROM chatbot WHERE user_id = ? AND query = ?",
                (user_id, query)
            )
            row = cursor.fetchone()
            if row:
                logger.info("📦 Cache hit. Returning cached response.")
                return JSONResponse(content={"response": row[0], "cached": True})

        # === Step 2: Semantic Retrieval ===
        logger.debug("🔎 Performing semantic retrieval using PathRAG.")
        nodes = pathrag.retrieve_nodes(query=query, top_k=top_k)
        paths = pathrag.prune_paths(nodes=nodes, max_hops=max_hop)

        if not paths:
            logger.warning("No valid paths found for query: %s", query)
            return JSONResponse(
                status_code=200,
                content={"message": "[!] No valid paths found. Try lowering prune_thresh or increasing max_hops."}
            )

        scored_paths = pathrag.score_paths(paths)
        logger.info("Retrieved and scored %d semantic paths.", len(scored_paths))

        # === Step 3: Prompt Generation ===
        logger.debug("🧠 Generating prompt from top paths.")
        prompt_chunks = [ctx["chunk"] for path in scored_paths for ctx in path if "chunk" in ctx]
        prompt_template = PromptOllama()
        prompt = prompt_template.prompt(query=query, retrieval_context=prompt_chunks)

        # === Step 4: Generate LLM Response ===
        logger.debug("💬 Calling Ollama LLM to generate response.")
        llm_response = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens
        )

        if not llm_response or llm_response.startswith("[ERROR]"):
            logger.error("🚨 LLM failed to generate a valid response.")
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid response.")

        logger.info("✅ LLM response generated successfully.")

        # === Step 5: Store in Cache/Log ===
        for idx, chunk_text in enumerate(prompt_chunks):
            success = insert_chatbot_entry(
                conn=conn,
                user_id=user_id,
                query=query,
                llm_response=llm_response,
                retrieval_context=chunk_text,
                retrieval_rank=idx + 1
            )
            if not success:
                logger.warning("⚠️ Failed to store chatbot entry for chunk #%d", idx + 1)

        return JSONResponse(content={"response": llm_response, "cached": False})

    except HTTPException as http_err:
        logger.warning("❌ HTTPException: %s", http_err.detail)
        raise

    except Exception as e:
        logger.exception("💥 Unexpected error in chatbot route: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chatbot operation."
        )
