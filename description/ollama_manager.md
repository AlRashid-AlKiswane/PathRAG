## 🧠 Ollama Manager

A robust **Ollama LLM system controller** for orchestrating local LLM workflows via the [Ollama CLI](https://ollama.com). This module wraps system-level operations to start the Ollama server, verify health, manage models, and launch LLMs interactively or in the background.

---

### 📂 File

```
src/utils/ollama_manager.py
```

---

### ✅ Features

* ✅ Dependency validation (`ollama`, `systemctl`, `curl`, `pgrep`)
* ✅ Automatic server detection and (re)start
* ✅ Model pulling via `ollama pull <model>`
* ✅ Interactive or background model execution
* ✅ Robust logging with emoji indicators for clarity

---

### 🧠 Class: `OllamaManager`

```python
OllamaManager()
```

#### Attributes

| Name         | Type  | Description                                                |
| ------------ | ----- | ---------------------------------------------------------- |
| `server_url` | `str` | Ollama server base URL (default: `http://localhost:11434`) |

---

### 🔧 Method: `_run_command`

```python
_run_command(cmd: str, check: bool = True, capture_output: bool = False) -> Optional[subprocess.CompletedProcess]
```

Executes a shell command with robust logging and error handling.

#### Args

* `cmd`: Shell command string
* `check`: Raise error on non-zero exit status (default: True)
* `capture_output`: Capture and return stdout/stderr (default: False)

#### Returns

* `CompletedProcess` or `None` on failure

---

### 🧪 Method: `check_dependencies`

```python
check_dependencies() -> bool
```

Checks for system prerequisites: `ollama`, `systemctl`, `curl`, and `pgrep`.

#### Returns

* `True` if all are found; otherwise `False`.

---

### 🔎 Method: `is_server_running`

```python
is_server_running() -> bool
```

Uses `pgrep` to check if the Ollama server is currently running.

---

### 🚀 Method: `start_server`

```python
start_server() -> bool
```

Starts the Ollama server. Tries:

1. `systemctl --user start ollama`
2. Fallback: `ollama serve &`

---

### 🩺 Method: `health_check`

```python
health_check() -> bool
```

Sends a health probe using `curl` to verify the Ollama server is live.

---

### 📥 Method: `pull_model`

```python
pull_model(model_name: str) -> bool
```

Pulls the specified model from Ollama's registry.

---

### 💬 Method: `run_model`

```python
run_model(model_name: str) -> bool
```

Launches the specified model in interactive mode (`ollama run <model>`), waits for user exit (`Ctrl+C`).

---

### 🔁 Method: `execute_workflow`

```python
execute_workflow(model_name: str)
```

Main workflow orchestration. Steps:

1. Check dependencies
2. Start Ollama server
3. Pull the specified model
4. Launch model in **background**

---

### 💻 CLI Usage

```bash
python ollama_manager.py <model_name>
```

#### Example

```bash
python ollama_manager.py llama2
```

> On first run, models will be downloaded and cached locally.

---

### 🔗 CLI Output Example

```plaintext
Ollama Manager - Model Runner
Usage: python ollama_manager.py <model_name>

Available models (examples):
  - llama2
  - mistral
  - codellama
```

---

### ⚠ Error Handling

| Type                            | Raised From          | Description                    |
| ------------------------------- | -------------------- | ------------------------------ |
| `subprocess.CalledProcessError` | `_run_command`       | Non-zero exit codes            |
| `FileNotFoundError`             | `check_dependencies` | Missing binaries               |
| `Exception`                     | All methods          | General catch-all with logging |