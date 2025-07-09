## 📁 File Upload API

This module defines an endpoint to handle secure **multi-file uploads**, including type validation, filename sanitization, and structured file organization. It is designed to support LightRAG’s document ingestion pipeline.

---

### 📂 File

```
src/routes/upload_route.py
```

---

### 🚀 Endpoint

```http
POST /api/v1/files/multi/
```

---

### ✅ Features

* Upload **multiple files** in one request
* Validates file type based on configured extension list
* Sanitizes and shortens directory names (max 30 characters)
* Automatically creates per-file subdirectories (or custom via `dir_name`)
* Prevents file collisions by generating **unique filenames**
* Detailed upload report: success/failure per file

---

### 🔧 Parameters

| Name       | Type                | Required | Default | Description                                                 |
| ---------- | ------------------- | -------- | ------- | ----------------------------------------------------------- |
| `files`    | `List[UploadFile]`  | ✅ Yes    | N/A     | Files to be uploaded. Multipart form required.              |
| `dir_name` | `str` (query param) | ❌ No     | None    | Custom directory name for all files. Defaults to file stem. |

---

### 📥 Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/files/multi/" \
  -F "files=@example.pdf" \
  -F "files=@data.txt" \
  -F "dir_name=my_uploads"
```

---

### 📤 Example Response

```json
{
  "success_count": 2,
  "failed_count": 0,
  "details": [
    {
      "original_name": "example.pdf",
      "saved_path": "/path/to/my_uploads/example_20250709_141212.pdf",
      "directory": "my_uploads",
      "size": "54.3 KB",
      "status": "success"
    },
    {
      "original_name": "data.txt",
      "saved_path": "/path/to/my_uploads/data_20250709_141212.txt",
      "directory": "my_uploads",
      "size": "3.2 KB",
      "status": "success"
    }
  ]
}
```

---

### 🛡 Error Handling

| Status Code | Reason                         |
| ----------- | ------------------------------ |
| `400`       | No files provided              |
| `207`       | Partial success or mixed files |
| `500`       | Unexpected error per file      |

---

### 🔐 File Validation

* Allowed extensions come from: `Settings.FILE_TYPES`
* Directories stored under: `Settings.DOC_LOCATION_STORE`
* Filenames sanitized and deduplicated via `generate_unique_filename()`

---

### 🧩 Dependencies

* `generate_unique_filename()`: Appends timestamp to avoid collisions
* `get_size()`: Returns human-readable file size
* `setup_logging()`: Central logger setup
* `Settings`: Uses `.env` or config store to control paths/extensions