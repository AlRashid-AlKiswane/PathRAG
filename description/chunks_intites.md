## 🏷️ Chunks → NER Extraction API

This FastAPI module defines an endpoint to extract **Named Entities** from text chunks stored in your database. It leverages a HuggingFace-powered NER model and populates the `ner_entities` table for downstream use in graph-based RAG pipelines.

---

### 📁 File

```
src/routes/chunks_intites.py
```

---

### 🚀 Endpoint

```http
POST /api/v1/ner
```

---

### ✅ Features

* Named Entity Recognition via a HuggingFace-based `NERModel` class
* Pulls chunks from any SQLite table (default: `chunks`)
* Inserts recognized entities into `ner_entities` table with:

  * Entity text
  * Type (e.g., PERSON, ORG, DATE)
  * Character offsets
  * Confidence score
  * Source chunk text
* Multilingual support (model-dependent)
* Full structured logging
* Defensive error handling

---

### 🔧 Parameters

| Name         | Type        | Required | Default                       | Description                                |
| ------------ | ----------- | -------- | ----------------------------- | ------------------------------------------ |
| `columns`    | `List[str]` | No       | `["id", "chunk", "dataName"]` | Columns to retrieve from the source table  |
| `table_name` | `str`       | No       | `"chunks"`                    | Name of the SQLite table containing chunks |

> Dependencies (DB connection and model) are auto-injected using FastAPI's `Depends()` mechanism.

---

### 📥 Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/ner"
```

With custom source table:

```bash
curl -X POST "http://localhost:8000/api/v1/ner?table_name=my_chunks"
```

---

### 📤 Example Response

```json
{
  "message": "NER extraction and insertion completed successfully.",
  "chunks_processed": 17,
  "entities_inserted": 89
}
```

---

### 🧠 Processing Pipeline

1. **Retrieve Chunks**: Pulls records with specified columns (must include `chunk` and `id`)
2. **Filter Invalid Rows**: Skips records missing chunk content
3. **NER Inference**: For each valid chunk, calls `NERModel.predict()`
4. **Entity Insertion**: Writes extracted entities to `ner_entities` table
5. **Tracks Failures**: Logs but skips failed insertions or model calls

---

### 🛡 Error Handling

| Status Code | Condition                              |
| ----------- | -------------------------------------- |
| `400`       | Invalid input or schema mismatch       |
| `500`       | Internal server error or model failure |

---

### 🧩 Dependencies

* [`pull_from_table()`](../db/pull_from_table.py): Loads chunk data from SQLite
* [`insert_ner_entity()`](../db/insert_ner_entity.py): Inserts entity data into `ner_entities`
* [`NERModel`](../llms_providers/ner_model.py): DistilBERT-style NER model with `.predict(text)` interface
