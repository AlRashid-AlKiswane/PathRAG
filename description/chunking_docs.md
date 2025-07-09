## ✂️ Document Chunking API

This module defines the route responsible for **splitting documents into chunks** and **inserting them into the database** for later retrieval.

---

### 📁 File

```
src/routes/chunking_route.py
```

### 🚀 Endpoint

```http
POST /api/v1/chunk/
```

---

### ✅ Features

* Accepts:

  * A **single file path** for targeted processing
  * A **directory name** to batch-process all files inside `assets/docs/<dir_file>`
* Performs text chunking using the `chunking_docs()` utility
* Stores chunks into the `chunks` table of a SQLite database
* Supports optional reset of the `chunks` table
* Gracefully handles:

  * Missing files/directories
  * Input validation errors

---

### 🔧 Query Parameters

| Name          | Type      | Required | Description                                                    |
| ------------- | --------- | -------- | -------------------------------------------------------------- |
| `file_path`   | `string`  | optional | Full path to a single document to be chunked                   |
| `dir_file`    | `string`  | optional | Subdirectory inside `assets/docs` for batch chunking           |
| `reset_table` | `boolean` | optional | If `true`, clears the existing `chunks` table before insertion |

> **Note:** Either `file_path` or `dir_file` must be provided, but not both.

---

### 📥 Example Request (Single File)

```bash
curl -X POST "http://localhost:8000/api/v1/chunk/?file_path=/full/path/to/file.txt"
```

### 📥 Example Request (Directory)

```bash
curl -X POST "http://localhost:8000/api/v1/chunk/?dir_file=tech_articles&reset_table=true"
```

---

### 📤 Example Response

```json
{
  "message": "Chunking successful",
  "total_chunks": 58
}
```

---

### 🧠 Internal Logic

1. **Optional Table Reset**: Clears the `chunks` table if `reset_table=True`
2. **File Mode**:

   * Verifies file exists
   * Uses `chunking_docs()` to split the file
   * Inserts each chunk into the DB
3. **Directory Mode**:

   * Resolves subdirectory path inside `assets/docs/<dir_file>`
   * Iterates over files, applies chunking, and stores all chunks
4. **Logs All Steps**: For traceability and debugging
5. **Returns**: A summary with total inserted chunks

---

### 🛡 Error Responses

| Status Code | Reason                                      |
| ----------- | ------------------------------------------- |
| 400         | Neither `file_path` nor `dir_file` provided |
| 404         | Provided file or directory not found        |
| 500         | Internal error during chunking/insertion    |

---

### 📌 Dependencies

* [`chunking_docs()`](../controllers/chunking_controller.py)
* [`insert_chunk()`](../db/insert_chunk.py)
* [`clear_table()`](../db/table_utils.py)
* SQLite3 database connection via FastAPI dependency