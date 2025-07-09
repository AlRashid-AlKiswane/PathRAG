## 🔍 Live Retrieval API

This module implements a **live semantic retrieval endpoint** combining both **vector similarity** and **entity-level filtering**.

### 📁 File

```
src/routes/live_retrieval.py
```

### 🚀 Endpoint

```http
POST /api/v1/retrieval
```

---

### ✅ Features

* **Hybrid Retrieval**: Combines:

  * `FaissRAG`: for dense vector similarity search using FAISS.
  * `EntityLevelFiltering`: for filtering based on Named Entity Recognition (NER).
* **Flexible Modes**: Combine results using:

  * `intersection`: overlap of both methods.
  * `union`: merge both results.
  * `faiss_only`: vector search only.
  * `entity_only`: entity filter only.
* **Pluggable Components**: Uses FastAPI's dependency injection for modularity.
* **Efficient Embedding**: Uses `HuggingFaceModel` for real-time embedding generation.
* **Robust Logging**: Detailed structured logs for debugging and observability.
* **Safe Error Handling**: All key operations wrapped in fail-safe error boundaries.

---

### 🔧 Query Parameters

| Name    | Type     | Description                                                                          |
| ------- | -------- | ------------------------------------------------------------------------------------ |
| `query` | `string` | User's natural language question or topic                                            |
| `top_k` | `int`    | Number of top chunks to retrieve (1 to 10, default: 3)                               |
| `mode`  | `string` | Result combination strategy:<br>`intersection`, `union`, `faiss_only`, `entity_only` |

---

### 📥 Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/retrieval?query=What is AI?&top_k=5&mode=intersection"
```

---

### 📤 Example Response (JSON)

```json
{
  "query": "What is AI?",
  "top_k": 5,
  "mode": "intersection",
  "results": [
    {
      "chunk_id": "chunk_123",
      "text": "Artificial Intelligence (AI) is the simulation of human intelligence..."
    },
    ...
  ]
}
```

---

### 🧠 Internal Flow

1. **Embed Query** → Converts query string to dense vector using `HuggingFaceModel`.
2. **FAISS Retrieval** → Finds nearest vectors via `FaissRAG`.
3. **Entity Retrieval** → Runs NER and finds relevant chunks via `EntityLevelFiltering`.
4. **Combine Results** → Based on selected mode, intelligently merges results.
5. **Return** → Structured JSON response.

---

### 🛡 Error Handling

* 400 for empty/invalid query
* 500 for:

  * Embedding generation failure
  * FAISS/Entity lookup failure
  * Result merging issues

---

### 📌 Dependencies

* [`FaissRAG`](../rag/faiss_rag.py)
* [`EntityLevelFiltering`](../rag/entities_level_filtering.py)
* [`HuggingFaceModel`](../llms_providers/embedding_model.py)
