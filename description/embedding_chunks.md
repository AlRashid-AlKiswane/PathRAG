## 🧠 Chunks → Embeddings API

This module defines a FastAPI endpoint that reads chunked text from the database, computes embeddings using a HuggingFace model, and stores the resulting vectors in the `embed_vector` table.

---

### 📁 File

```
src/routes/embedding_chunks_route.py
```

### 🚀 Endpoint

```http
POST /api/v1/chunks_to_embeddings
```

---

### ✅ Features

* Pulls text chunks from any SQLite table (default: `chunks`)
* Uses a HuggingFace model to generate high-dimensional embedding vectors
* Stores vectors into `embed_vector` table with associated metadata
* Skips invalid/empty chunks
* Robust error handling:

  * Database errors
  * Embedding failures
  * Timeouts or cancellations
* Structured logging for traceability

---

### 🔧 Parameters (via Query or JSON)

| Name         | Type        | Required | Default                       | Description                               |
| ------------ | ----------- | -------- | ----------------------------- | ----------------------------------------- |
| `columns`    | `List[str]` | No       | `["id", "chunk", "dataName"]` | Columns to retrieve from the source table |
| `table_name` | `str`       | No       | `"chunks"`                    | SQLite table name to read chunks from     |

> Note: This route uses FastAPI **dependency injection** to access the SQLite connection and embedding model.

---

### 📥 Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/chunks_to_embeddings"
```

Or with a custom table:

```bash
curl -X POST "http://localhost:8000/api/v1/chunks_to_embeddings?table_name=my_chunks"
```

---

### 📤 Example Response

```json
{
  "message": "Chunks successfully embedded and stored.",
  "chunks_processed": 42
}
```

---

### 🧠 Internal Logic

1. **Retrieve Rows**: Pulls records from the specified table with `id`, `chunk`, and optional metadata
2. **Filter Valid Chunks**: Skips records with missing or empty `chunk` values
3. **Loop & Embed**:

   * Generates an embedding for each valid chunk using `HuggingFaceModel.embed_texts`
   * Serializes the vector using `json.dumps()`
   * Inserts it into the `embed_vector` table using `insert_embed_vector()`
4. **Handles Errors**:

   * Logs and skips over individual failures (chunk-level)
   * Halts on database failures (table-level)

---

### 🛡 Error Handling

| Status Code | Condition                                                         |
| ----------- | ----------------------------------------------------------------- |
| `500`       | Database error, embedding crash, invalid table or no chunks found |
| `504`       | Embedding process timed out                                       |
| `500`       | Embedding process cancelled or unexpected failure                 |

---

### 🧩 Dependencies

* [`pull_from_table()`](../db/pull_from_table.py): Retrieves chunk records from SQLite
* [`insert_embed_vector()`](../db/insert_embed_vector.py): Inserts embedding into vector store
* [`HuggingFaceModel`](../llms_providers/huggingface.py): Embedding model wrapper
* Uses dependency-injected database connection and model instance
