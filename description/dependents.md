## 🧰 `src/dependents.py`

### 🔗 App State Dependency Access Utilities

This module centralizes access to shared application resources stored in `FastAPI.app.state`, including the:

* SQLite database connection
* LLM (Ollama)
* Embedding model (HuggingFace)
* Named Entity Recognition (NER) model
* FAISS-based semantic retriever (`FaissRAG`)
* Entity-level filter (`EntityLevelFiltering`)

---

### 📄 Purpose

These accessors simplify and **standardize retrieval** of singleton resources in route handlers or service layers, reducing boilerplate and ensuring robust logging and consistent error messages when a resource is unavailable or misconfigured.

---

### ✅ Usage Example (With FastAPI Dependency Injection)

```python
from fastapi import APIRouter, Request, Depends
from src import get_llm, get_db_conn

router = APIRouter()

@router.get("/ask")
def ask_rag_question(request: Request, llm=Depends(get_llm)):
    # Use OllamaModel instance safely
    return {"status": "LLM is alive"}
```

---

### 🔐 Error Handling

| Exception Type  | Status Code                 | Trigger Condition                   |
| --------------- | --------------------------- | ----------------------------------- |
| `HTTPException` | `503` Service Unavailable   | Resource not present in `app.state` |
| `HTTPException` | `500` Internal Server Error | Unexpected errors in access logic   |

All errors are:

* Fully logged with `logger.exception()` for unexpected crashes
* Explained clearly via FastAPI-compatible HTTP errors

---

### 🧱 Available Functions

| Function                              | Returns                | Description                                              |
| ------------------------------------- | ---------------------- | -------------------------------------------------------- |
| `get_db_conn(request)`                | `sqlite3.Connection`   | Returns the SQLite connection from app state             |
| `get_llm(request)`                    | `OllamaModel`          | Retrieves the LLM model used for generation              |
| `get_embedding_model(request)`        | `HuggingFaceModel`     | Retrieves the transformer-based embedding model          |
| `get_ner_model(request)`              | `NERModel`             | Returns the NER model for entity extraction              |
| `get_faiss_rag(request)`              | `FaissRAG`             | Retrieves the FAISS-based semantic RAG instance          |
| `get_entity_level_filtering(request)` | `EntityLevelFiltering` | Returns the filtering utility for entity-based retrieval |

---

### 🧼 Design Considerations

* All functions are **safe to use with FastAPI's `Depends()`** system
* Each access function includes logging at `DEBUG` and `ERROR` levels
* Failures are surfaced to clients in a **user-safe but actionable** format