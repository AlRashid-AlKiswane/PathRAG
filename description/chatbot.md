## 📡 `/api/v1/chatbot` — Chatbot API Endpoint

### 🔍 **Purpose**

This route handles chatbot interactions using a **Retrieval-Augmented Generation (RAG)** pipeline. It combines:

* **Vector-based retrieval** with FAISS,
* **Entity-level filtering** to refine context,
* **Local LLM inference** (via Ollama),
* **Prompt engineering**, and
* **Optional response caching** for performance.

---

### 🧠 **Workflow Overview**

1. **Input Parsing**
   Accepts a `POST` request with a JSON payload matching the `Chatbot` schema:

   ```json
   {
     "query": "...",
     "top_k": 5,
     "temperature": 0.7,
     "max_new_tokens": 200,
     "max_input_tokens": 1000,
     "mode_retrieval": "hybrid",  // or "semantic"/"entity"
     "user_id": "user123",
     "cache": true
   }
   ```

2. **Cache Check**
   If `cache=true`, the route first checks the SQLite database (`chatbot` table) for a previously stored response for the same `user_id` and `query`.

3. **Context Retrieval**
   Uses a dual-level approach:

   * **Semantic similarity** (via embedding vectors)
   * **Entity filtering** (NER-driven filtering for relevance)

4. **Prompt Generation**
   Creates a prompt using the retrieved context chunks and the query. It uses `PromptOllama` to format the final prompt string.

5. **LLM Inference**
   Passes the prompt to a local LLM (Ollama) with configurable generation parameters (`temperature`, `max_new_tokens`, etc.).

6. **Result Storage**
   If generation succeeds, the chatbot's full context and answer are stored in the database, chunk by chunk, with ranking metadata.

7. **Response**
   Returns a JSON payload:

   ```json
   {
     "response": "LLM answer here...",
     "cached": false
   }
   ```

---

### 🔐 **Error Handling**

* `404 Not Found`: If no relevant context is retrieved.
* `500 Internal Server Error`: For LLM or processing failures.
* Caught exceptions are logged with full tracebacks.

---

### 🧩 **Dependencies Injected**

| Dependency               | Description                               |
| ------------------------ | ----------------------------------------- |
| `conn` (SQLite conn)     | Database connection for cache and storage |
| `llm`                    | Ollama LLM instance                       |
| `faiss_rag`              | FAISS-based semantic retriever            |
| `entity_level_filtering` | Entity-based filtering module             |
| `embed_model`            | Sentence transformer model for embeddings |

---

### 🗃️ **Logged Events**

* Info logs for request/response metadata
* Debug logs for cache hits, retrieval results
* Warnings for storage or logic failures
* Errors with full stack traces on unexpected failures

---

### ✅ **Example cURL Request**

```bash
curl -X POST http://localhost:8000/api/v1/chatbot \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum computing?",
    "top_k": 3,
    "temperature": 0.6,
    "max_new_tokens": 150,
    "max_input_tokens": 1000,
    "mode_retrieval": "hybrid",
    "user_id": "user42",
    "cache": true
  }'
```

# **Chatbot route pipeline**
```mermaid
graph TD
    A[Input Parsing<br/>(Request Body: Chatbot)] --> B{Cache Enabled?}
    B -- Yes --> C[Check DB for Cached Response]
    C -- Hit --> H[Return Cached Response]

    B -- No or Miss --> D[Dual-Level Retrieval<br/>(FAISS + Entity Filter)]
    D --> E[Prompt Generation<br/>(PromptOllama)]
    E --> F[LLM Inference<br/>(OllamaModel.generate)]
    F --> G[Store Result in DB<br/>(Query + Context + Response)]
    G --> H[Return Response JSON<br/>{"response": ..., "cached": False}]
```
