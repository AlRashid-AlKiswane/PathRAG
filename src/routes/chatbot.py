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
    faiss_rag: FaissRAG = Depends(get_faiss_rag),
    entity_level_filtering: EntityLevelFiltering = Depends(get_entity_level_filtering),
    embed_model: HuggingFaceModel = Depends(get_embedding_model)
):
    """
    Handles chatbot requests using retrieval-augmented generation (RAG) and a local LLM.
    Supports entity-level and semantic filtering. Uses cache if enabled.

    Args:
        body (Chatbot): Chatbot request parameters including query, top_k, temperature, etc.

    Returns:
        JSONResponse: Response with the generated answer and cache status.
    """
    try:
        query = body.query
        top_k = body.top_k
        temperature = body.temperature
        max_new_tokens = body.max_new_tokens
        max_input_tokens = body.max_input_tokens
        mode_retrieval = body.mode_retrieval
        user_id = body.user_id
        cache = body.cache

        logger.info("🚀 Chatbot request | user_id='%s' | query='%s' | cache=%s", user_id, query, cache)

        # === Cache Check ===
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

        # === Retrieval Phase ===
        logger.debug("📡 Performing retrieval (mode='%s') for top_k=%d", mode_retrieval, top_k)
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

        # === Prompt Construction ===
        context_chunks = [ctx["chunk"] for ctx in retrieval_result]
        prompt_template = PromptOllama()
        prompt = prompt_template.prompt(query=query, retrieval_context=context_chunks)

        # === LLM Generation ===
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

        # === Store Results ===
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
