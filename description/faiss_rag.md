## 🔍 `faiss_rag.py` – FAISS-based Semantic Retrieval

This module implements a **lightweight semantic retriever** using [FAISS](https://github.com/facebookresearch/faiss) and vector embeddings stored in a SQLite database. It provides efficient top-𝑘 nearest neighbor retrieval using **L2 distance** and includes comprehensive logging for observability and debugging.

---

### 📂 File

```
src/rag/faiss_rag.py
```

---

### ✅ Features

* Fetches precomputed embeddings from `embed_vector` table
* Dynamically builds an in-memory FAISS index
* Supports semantic search with top-k nearest chunk results
* Full logging for every step: vector loading, FAISS search, DB access

---

### 🧠 Class: `FaissRAG`

```python
FaissRAG(conn: sqlite3.Connection)
```

#### Attributes

| Name   | Type                 | Description              |
| ------ | -------------------- | ------------------------ |
| `conn` | `sqlite3.Connection` | Active SQLite connection |

---

### 🔄 Method: `_fetch_embedding_vector`

```python
_fetch_embedding_vector() -> Tuple[np.ndarray, List[int]]
```

#### Description:

Loads `chunk_id` and `embedding` vectors from `embed_vector` table. Parses JSON-encoded embedding blobs into NumPy arrays.

#### Returns:

* `vectors_embed`: `np.ndarray` of shape `(n_chunks, dim)`
* `chunk_ids`: `List[int]` of corresponding IDs

#### Raises:

* `ValueError` if no valid embeddings found

---

### 🛠 Method: `_build_faiss_index`

```python
_build_faiss_index(vectors_embedding: np.ndarray) -> faiss.IndexFlatL2
```

#### Description:

Creates a **FAISS flat L2 index** and populates it with `vectors_embedding`.

#### Returns:

* `faiss.IndexFlatL2` object

---

### 🔍 Method: `semantic_retrieval`

```python
semantic_retrieval(embed_query: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]
```

#### Description:

Main interface for retrieving top-k most similar chunks to a query embedding.

#### Args:

* `embed_query`: 1D NumPy array representing the query vector
* `top_k`: Number of most similar chunks to return

#### Returns:

* List of dicts: `{ "id": int, "chunk": str }`

#### Raises:

* `ValueError` if embedding shape is invalid
* `Exception` on index/search failure

---

### 📤 Method: `_fetch_chunk_retrieval`

```python
_fetch_chunk_retrieval(indices: List[int], chunk_ids: List[int]) -> List[Dict[str, Any]]
```

#### Description:

Takes indices from FAISS and fetches corresponding chunks from `chunks` table using their IDs.

#### Returns:

* List of dicts: `{ "id": int, "chunk": str }`

---

### 📦 Database Schema Dependencies

#### Table: `embed_vector`

```sql
CREATE TABLE embed_vector (
    chunk_id INTEGER,
    embedding TEXT -- JSON-encoded float32 array
);
```

#### Table: `chunks`

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    chunk TEXT
);
```

---

### 📥 Example Usage

```python
from src.retrievers.faiss_rag import FaissRAG
from src.utils import get_db_conn

conn = get_db_conn()
retriever = FaissRAG(conn)

query_vector = get_query_embedding("What is GPT-4?")
top_results = retriever.semantic_retrieval(query_vector, top_k=5)
```

---

### ⚠ Error Handling

| Error Type      | Source                    | Description                          |
| --------------- | ------------------------- | ------------------------------------ |
| `ValueError`    | `_fetch_embedding_vector` | No vectors or shape mismatch         |
| `Exception`     | `semantic_retrieval`      | FAISS failure or unexpected issue    |
| `sqlite3.Error` | `_fetch_chunk_retrieval`  | SQL error while fetching text chunks |
