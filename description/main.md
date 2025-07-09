## 🚀 Graph-RAG FastAPI Entry Point

The **main application module** for the Graph-RAG system. It bootstraps the FastAPI server, connects to the SQLite database, initializes machine learning models, and mounts all core routes for the RAG system.

---

### 📂 File

```
src/main.py
```

---

### 📌 Responsibilities

* ✅ Configure base directory for relative imports
* ✅ Initialize structured logging
* ✅ Set up SQLite connection and tables
* ✅ Load core LLM, embedding, and NER models
* ✅ Instantiate FAISS and entity-level retrieval systems
* ✅ Register FastAPI routers for each processing stage

---

### 📦 App Overview

```python
app = FastAPI(
    title="Graph-RAG API",
    version="1.0.0",
    description="A lightweight RAG system with semantic and entity-level filtering"
)
```

#### Mounted Routers:

| Route Prefix | Module                   | Description                               |
| ------------ | ------------------------ | ----------------------------------------- |
| `/upload`    | `upload_route`           | Handles document uploads                  |
| `/chunk`     | `chunking_route`         | Splits documents into vectorizable chunks |
| `/embed`     | `embedding_chunks_route` | Embeds chunks using HuggingFace models    |
| `/ner`       | `ner_route`              | Extracts named entities from content      |
| `/retrieve`  | `live_retrieval_route`   | Performs semantic/entity RAG retrieval    |

---

### 🔄 Lifespan Handler: `lifespan(app: FastAPI)`

Manages the application lifecycle with an `asynccontextmanager`.

#### ✅ Startup Tasks:

1. **Connect to SQLite** (`app.state.conn`)
2. **Initialize Tables**: `chunks`, `embed_vector`, `ner_entities`
3. **Load Models**:

   * `OllamaModel` (`app.state.llm`)
   * `HuggingFaceModel` for embeddings
   * `NERModel` for entity recognition
4. **Instantiate Core Components**:

   * `FaissRAG` semantic retriever
   * `EntityLevelFiltering` module

#### 🧼 Shutdown Tasks:

* Gracefully closes the SQLite connection

---

### ⚠️ Error Handling

| Failure Type               | Handled In      | Behavior                   |
| -------------------------- | --------------- | -------------------------- |
| Import or path errors      | Top-level block | Logs and exits             |
| DB connection/init failure | `lifespan`      | Logs critical error, exits |
| Model loading failure      | `lifespan`      | Logs critical error, exits |
| RAG component failures     | `lifespan`      | Logs and exits             |
| Router registration issues | Global block    | Logs and exits             |

---

### 📌 Logging Highlights

* ✅ Uses `setup_logging()` for structured output
* 🧠 Logs lifecycle stages with emoji indicators
* 🔍 Logs full traceback for all startup failures
* ⚠ Includes fallback logging if DB close fails

---

### 🧪 Local Development

To run the server locally:

```bash
uvicorn src.main:app --reload
```

You must have the following prerequisites:

* ✅ SQLite accessible in the `data/` directory
* ✅ Models configured in `.env` via `Settings`
* ✅ Ollama server running if using `OllamaModel`

---

### 🔧 Configuration Environment Variables (`.env`)

| Variable          | Description                                   |
| ----------------- | --------------------------------------------- |
| `OLLAMA_MODEL`    | Model identifier for Ollama (e.g. `"llama2"`) |
| `EMBEDDING_MODEL` | Embedding model for chunk vectorization       |
| `NER_MODEL`       | NER model for entity extraction               |
