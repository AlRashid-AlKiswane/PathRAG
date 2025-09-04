# Ollama Provider Module (`ollama_provider.py`)

---

## Overview

The **Ollama Provider** module provides a robust interface for interacting with **local Ollama LLMs** using HTTP requests.  

It is designed to enable **reliable text generation** within **RAG pipelines, chatbot systems, and server-side NLP services**, offering:

- Model initialization and warm-up  
- Prompt submission with retry logic  
- Automatic truncation of long inputs  
- Health checks and service availability verification  
- Detailed logging and exception handling  

This module ensures that large language models (LLMs) managed by Ollama are **ready, responsive, and reliable** for downstream applications.

---

## Dependencies

- `requests` → HTTP client for Ollama API  
- `time` → Wait intervals and warm-up delays  
- `src.utils.OllamaManager` → Trigger model loading workflows  
- `src.infra.setup_logging` → Structured logging for operational transparency  

---

## Class: `OllamaModel`

Provides an **enhanced interface** for interacting with Ollama-based LLMs.

---

### Initialization

```python
model = OllamaModel(model_name="gemma3:1b", base_url="http://localhost:11434")
````

* **Parameters:**

  * `model_name` *(str)* → Name of the Ollama model (default `"gemma3:1b"`)
  * `base_url` *(str)* → URL of the Ollama server (default `"http://localhost:11434"`)

* **Behavior:**

  * Executes model workflow using `OllamaManager`
  * Waits for model readiness via `_wait_for_model_ready()`
  * Initializes logger
  * Catches and logs critical exceptions during initialization

---

### Method: `_wait_for_model_ready(max_wait_time: int = 30) -> None`

Waits until the model is ready for requests.

* **Parameters:**

  * `max_wait_time` *(int)* → Maximum wait time in seconds (default 30)

* **Behavior:**

  * Periodically tests the model with a simple request
  * Logs INFO once the model is ready
  * Logs WARNING if the model is not ready after the timeout

---

### Method: `_make_request(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str`

Sends a request to the Ollama API and handles errors.

* **Parameters:**

  * `prompt` *(str)* → Prompt text
  * `max_tokens` *(int)* → Maximum number of tokens to generate
  * `temperature` *(float)* → Sampling temperature

* **Returns:**

  * Generated text string
  * Error string (e.g., `[ERROR] Request failed.`) in case of failure

* **Error Handling:**

  * Timeouts, connection errors, HTTP errors, and unexpected exceptions are caught and logged

---

### Method: `generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.1, system_message: Optional[str] = None, retry_count: int = 3, max_input_tokens: int = 1024) -> str`

Generates text using the Ollama model with **retry logic** and prompt safety.

* **Parameters:**

  * `prompt` *(str)* → User prompt
  * `max_new_tokens` *(int)* → Maximum tokens to generate in the response
  * `temperature` *(float)* → Sampling temperature
  * `system_message` *(Optional\[str])* → Optional system-level instructions
  * `retry_count` *(int)* → Number of retry attempts on failure
  * `max_input_tokens` *(int)* → Truncates input prompt for safety

* **Returns:**

  * Generated text string
  * `[ERROR]` prefixed string if generation fails

* **Behavior:**

  * Prepares prompt by combining system message and user input
  * Truncates input to `max_input_tokens`
  * Retries generation up to `retry_count` times if failures occur
  * Logs all critical steps, retries, and warnings

---

### Method: `is_available() -> bool`

Checks if the Ollama server is available.

* **Returns:**

  * `True` if the server responds to `/api/version`
  * `False` otherwise

---

### Method: `get_model_info() -> dict`

Retrieves information about the loaded Ollama model.

* **Returns:**

  * Dictionary containing model metadata
  * `{"error": "..."}` if retrieval fails

* **Behavior:**

  * Makes a GET request to `/api/show` endpoint
  * Logs errors if the request fails

---

## Usage Example

```python
from ollama_provider import OllamaModel

# Initialize Ollama client
model = OllamaModel(model_name="gemma3:1b")

# Generate text from a prompt
response = model.generate(
    prompt="Explain the difference between CPU and GPU.",
    system_message="You are a helpful assistant.",
    max_new_tokens=256,
    temperature=0.2
)

# Check if server is available
if model.is_available():
    info = model.get_model_info()
    print(info)
```

---

## Notes

* Ollama must be **installed and running locally** on the configured port.
* Automatically handles **prompt truncation**, **retry attempts**, and **empty responses**.
* Integrated **detailed logging** for debugging and operational transparency.
* Ideal for **RAG pipelines, chatbots, or NLP services** that rely on local LLM execution.

---

## Best Practices

1. **Retry Configuration:** Adjust `retry_count` based on network stability and model load.
2. **Prompt Safety:** Limit prompt length using `max_input_tokens` to prevent server overload.
3. **System Messages:** Use `system_message` to guide the model’s behavior consistently.
4. **Server Health Check:** Use `is_available()` before sending heavy workloads.
5. **Logging:** Ensure logger is configured to monitor model initialization and generation issues.

---

## Common Errors

| Error                                               | Cause                               | Resolution                                   |
| --------------------------------------------------- | ----------------------------------- | -------------------------------------------- |
| `[ERROR] Empty prompt.`                             | Empty or whitespace-only input      | Provide a valid prompt                       |
| `[ERROR] Request timed out.`                        | Network issues or overloaded server | Check server status and network connectivity |
| `[ERROR] Connection error.`                         | Ollama server not running           | Start Ollama server at configured port       |
| `[ERROR] Server error - model not ready.`           | Model not loaded                    | Wait for model warm-up and retries           |
| `[ERROR] Text generation failed after all retries.` | Persistent generation failure       | Inspect logs; reduce prompt length or tokens |

---

## Conclusion

The `ollama_provider.py` module provides a **robust, reliable interface** for local Ollama LLMs.

It ensures:

* Safe and efficient prompt submission
* Retry logic for transient failures
* Automatic model warm-up and readiness checks
* Detailed logging for production use

This module is a **core component** for building **local RAG systems, chatbots, and NLP services** that require **highly available LLMs**.

```
