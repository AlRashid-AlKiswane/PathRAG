## 🧠 `OllamaModel` – Enhanced Ollama-Based LLM Client

A high-level wrapper for interacting with **local LLMs via the Ollama API**, offering seamless server management, automatic model downloading, and robust text generation with retries, logging, and error handling. This client powers the text generation engine in the lightweight RAG system.

---

### 📂 File

```
src/llms_providers/ollama_model.py
```

---

### ✅ Features

* 🖥️ Auto-starts and manages the Ollama server via `OllamaManager`
* 📦 Auto-downloads required model on first use
* ⏳ Waits for model readiness with a test-prompt ping
* 🔁 Implements retry logic on request failure
* 🔍 Includes helper utilities like `is_available()` and `get_model_info()`
* ⚠️ Logs all stages of the request lifecycle with rich error context

---

### 🧠 Class: `OllamaModel`

```python
OllamaModel(model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434")
```

#### Args:

| Name         | Type  | Default                  | Description                        |
| ------------ | ----- | ------------------------ | ---------------------------------- |
| `model_name` | `str` | `"gemma3:1b"`            | Ollama-compatible model identifier |
| `base_url`   | `str` | `http://localhost:11434` | Base Ollama API endpoint           |

---

### 🔄 Method: `_wait_for_model_ready`

```python
_wait_for_model_ready(max_wait_time: int = 30) -> None
```

Waits until the model can respond to prompts by polling every 2 seconds. Uses a test message `"Hello"` to verify readiness.

---

### 📤 Method: `_make_request`

```python
_make_request(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str
```

Internal method to send a POST request to `/api/generate` with generation settings.

---

### 💬 Method: `generate`

```python
generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    system_message: Optional[str] = None,
    retry_count: int = 3
) -> str
```

Generates a response for the provided prompt with configurable temperature, token count, and retry logic.

#### Args:

| Name             | Type    | Description                                  |
| ---------------- | ------- | -------------------------------------------- |
| `prompt`         | `str`   | User message or query to generate a reply to |
| `max_tokens`     | `int`   | Max tokens to generate in the response       |
| `temperature`    | `float` | Sampling temperature                         |
| `system_message` | `str?`  | Optional system-level prompt prefix          |
| `retry_count`    | `int`   | Number of retry attempts on failure          |

#### Returns:

* `str`: Model response or error placeholder (`"[ERROR] ..."`)

---

### 🔍 Method: `is_available`

```python
is_available() -> bool
```

Performs a health check using `/api/version` to verify the Ollama server is live.

---

### 📊 Method: `get_model_info`

```python
get_model_info() -> dict
```

Fetches metadata about the currently loaded model using `/api/show`.

---

### 🧪 CLI Example (for testing)

```bash
python src/llms_providers/ollama_model.py
```

```plaintext
✅ Ollama service is available
Response: The capital of France is Paris.
```

---

### ⚠ Error Handling

| Exception Type                        | Source          | Description                                     |
| ------------------------------------- | --------------- | ----------------------------------------------- |
| `requests.exceptions.Timeout`         | `_make_request` | Timeout on API request                          |
| `requests.exceptions.ConnectionError` | `_make_request` | Ollama server unreachable                       |
| `requests.exceptions.HTTPError`       | `_make_request` | HTTP 4xx/5xx error, including model load issues |
| `Exception`                           | all methods     | Fallback for unexpected failures                |

---

### 📌 Dependencies

* Ollama installed locally (`ollama serve`)
* Model (e.g. `gemma3:1b`) available or downloadable via `ollama pull`
* Server endpoint live at `http://localhost:11434`