## 🧠 Entity-Level Filtering Module

This module enables **entity-aware retrieval** by extracting named entities from a user query and retrieving associated chunk texts (via their `source_text`) from a SQLite database. It supports partial text matching and exact type filtering.

---

### 📂 File

```
src/rag/entity_level_filtering.py
```

---

### ✅ Features

* Named Entity Recognition (NER) on user query
* Retrieves related `source_text` from `ner_entities` table
* Supports both **partial entity text** and **exact type match**
* Dedupe logic for unique chunk-level results
* Full error handling and logging for debugging and observability

---

### 🧠 Class: `EntityLevelFiltering`

```python
EntityLevelFiltering(conn: sqlite3.Connection, ner_model: NERModel)
```

#### Attributes

| Name        | Type                 | Description                                  |
| ----------- | -------------------- | -------------------------------------------- |
| `conn`      | `sqlite3.Connection` | SQLite DB connection                         |
| `ner_model` | `NERModel`           | HuggingFace-style NER model with `predict()` |

---

### 🔍 Method: `_ner_query`

```python
_ner_query(query: str) -> List[Dict[str, str]]
```

#### Description:

Uses the injected NER model to extract entities from the user query.

#### Returns:

* List of dicts, each containing:

  * `text`: entity surface string
  * `type`: entity type (e.g., PERSON, ORG)

---

### 🧾 Method: `_fetch_chunks_from_db`

```python
_fetch_chunks_from_db(text: str, entity_type: str, top_k: int) -> List[Dict[str, str]]
```

#### Description:

Fetches up to `top_k` chunks (`source_text`) that either:

* Partially match the `text` (`LIKE %text%`)
* Exactly match the `type`

#### Returns:

* List of dicts with structure: `{"chunk": <source_text>}`

---

### 🔄 Method: `entities_retrieval`

```python
entities_retrieval(query: str, top_k: int) -> List[Dict[str, str]]
```

#### Description:

Main interface. Performs:

1. NER on input query
2. Fetches associated chunks per entity (text or type)
3. Deduplicates chunks

#### Returns:

* List of unique chunk dicts: `{"chunk": <text>}`

---

### 📥 Example Usage

```python
from src.filters.entity_level_filtering import EntityLevelFiltering
from src.llms_providers import get_ner_model
from src import get_db_conn

conn = get_db_conn()
ner_model = get_ner_model()

entity_filter = EntityLevelFiltering(conn=conn, ner_model=ner_model)
result_chunks = entity_filter.entities_retrieval("Who runs OpenAI and what is GPT-4?", top_k=5)
```

---

### ⚠ Error Handling

| Type                    | Raised from             | Reason                             |
| ----------------------- | ----------------------- | ---------------------------------- |
| `RuntimeError`          | `_ner_query`            | Model failure or malformed output  |
| `sqlite3.DatabaseError` | `_fetch_chunks_from_db` | SQL syntax or DB corruption issues |
| `RuntimeError`          | `entities_retrieval`    | Any internal failure               |

---

### 🔗 Database Dependency

Relies on the table:

```sql
CREATE TABLE ner_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT,
    text TEXT,
    type TEXT,
    start INTEGER,
    end INTEGER,
    score REAL,
    source_text TEXT
);
```

* Matching is done using:

  * `WHERE text LIKE %text% OR type = ?`

