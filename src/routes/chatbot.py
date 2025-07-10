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
from src import (get_db_conn, get_llm,
                 get_faiss_rag,
                 get_embedding_model,
                 get_entity_level_filtering)

from src.rag import dual_level_retrieval, FaissRAG, EntityLevelFiltering
from src.prompt import PromptOllama

# Initialize logger and settings
logger = setup_logging()
app_settings: Settings = get_settings()

chatbot_route = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["Chatbot"],
    responses={404: {"description": "Not found"}}
)

@chatbot_route.post("", response_class=JSONResponse)
async def chatbot(
    query: str = Query(..., alias="quyer"),
    top_k: int = 3,
    temperature: float = Query(0.4, alias="tempreture"),
    max_new_tokens: int = 512,
    max_input_tokens: int = 1024,
    mode_retrieval: str = Query("union", alias="mode_retreval"),
    user_id: str = "exe-012e",
    cache: bool = Query(True, alias="cach"),
    conn: Connection = Depends(get_db_conn),
    llm: OllamaModel = Depends(get_llm),
    faiss_rag: FaissRAG = Depends(get_faiss_rag),
    entity_level_filtering: EntityLevelFiltering = Depends(get_entity_level_filtering),
    embed_model: HuggingFaceModel = Depends(get_embedding_model)
):
    """
    Handles a chatbot request using a local LLM model, retrieval-based context, and optional caching.

    If caching is enabled and a response for the same query/user exists in the database,
    it will return the cached response. Otherwise, it performs dual-level semantic/entity retrieval,
    generates a response with an LLM, stores it, and returns it.

    Args:
        query (str): User input query.
        top_k (int): Number of top chunks to retrieve for context.
        temperature (float): Sampling temperature for the LLM.
        max_new_tokens (int): Maximum number of tokens to generate.
        max_input_tokens (int): Maximum number of tokens allowed in the input prompt.
        mode_retrieval (str): Retrieval strategy ("union", "intersection", etc.).
        user_id (str): Unique identifier for the querying user.
        cache (bool): Whether to use cached responses from previous queries.
        conn (Connection): SQLite database connection.
        llm (OllamaModel): Injected LLM instance for generation.
        faiss_rag (FaissRAG): FAISS-based semantic retriever.
        entity_level_filtering (EntityLevelFiltering): Entity-based filtering module.
        embed_model (HuggingFaceModel): Embedding model for query/vector conversion.

    Returns:
        JSONResponse: JSON containing the LLM response and cache status.
    """
    try:
        logger.info("🚀 Chatbot request | user_id='%s' | query='%s' | cache=%s", user_id, query, cache)

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

        logger.debug("📡 Performing retrieval (mode='%s') for top_k=%d", mode_retrieval, top_k)

        # Run dual-level retrieval (semantic + entity)
        retrieval_result = dual_level_retrieval(
            embed_model=embed_model,
            entity_level_filtering=entity_level_filtering,
            faiss_rag=faiss_rag,
            mode=mode_retrieval,
            query=query,
            top_k=top_k
        )

        if not retrieval_result:
            logger.warning("⚠️ No relevant context found for query: '%s'", query)
            raise HTTPException(status_code=404, detail="No retrieval context found.")

        logger.debug("📚 Retrieved %d context chunks.", len(retrieval_result))

        # Construct context prompt for generation
        context_chunks = [ctx["chunk"] for ctx in retrieval_result]
        prompt_templte = PromptOllama()
        prompt = prompt_templte.prompt(query=query, retrieval_context=context_chunks)

        # Generate response
        llm_response = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens
        )

        if not llm_response or llm_response.startswith("[ERROR]"):
            logger.error("🚨 LLM returned invalid response: %s", llm_response)
            raise HTTPException(status_code=500, detail="Failed to generate valid LLM response.")

        logger.info("✅ LLM response generated.")

        # Store all context chunks with their rank
        for idx, ctx in enumerate(retrieval_result):
            success = insert_chatbot_entry(
                conn=conn,
                user_id=user_id,
                query=query,
                llm_response=llm_response,
                retrieval_context=ctx["chunk"],
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
