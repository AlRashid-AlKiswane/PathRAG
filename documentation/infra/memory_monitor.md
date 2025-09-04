# Memory Monitor (`memory_monitor.py`)

---

## Overview

The **Memory Monitor** module provides a utility class, `MemoryMonitor`, for **tracking and enforcing memory usage limits** in Python applications.  

It is particularly useful in environments where **resource efficiency** and **system stability** are critical, such as:

- 🚀 Retrieval-Augmented Generation (RAG) pipelines  
- 📊 Data processing workflows  
- 🤖 Machine learning model serving  
- 🖥️ Long-running background services  

By integrating `MemoryMonitor`, developers can prevent **unexpected crashes due to out-of-memory (OOM) errors** and gain visibility into how memory usage changes during execution.

---

## Design Goals

1. **Simplicity** → A lightweight utility with minimal dependencies.  
2. **Safety** → Prevents silent memory overuse by logging warnings.  
3. **Observability** → Provides clear insight into memory growth.  
4. **Flexibility** → Works in any Python process with configurable limits.  
5. **Integration** → Easily pluggable into existing code with a context manager.

---

## Features

- ✅ Monitor **current process memory usage** (in GB)  
- ✅ Define **custom memory usage limits** (default = 4 GB)  
- ✅ Check if the process is **within the safe memory range**  
- ✅ Context manager (`memory_guard`) for automatic monitoring  
- ✅ Logs:
  - ⚠️ **Warnings** if memory exceeds the limit  
  - ℹ️ **Info messages** if memory usage grows but remains under the limit  

---

## Dependencies

- [psutil](https://pypi.org/project/psutil/) → Query process memory usage  
- [logging](https://docs.python.org/3/library/logging.html) → Structured logging  

Install dependencies:
```bash
pip install psutil
````

---

## Class: `MemoryMonitor`

### Purpose

The central class of this module. It **tracks the memory usage** of the current Python process and enforces a configurable limit.

---

### Attributes

* **`limit_bytes: int`**
  Memory usage threshold in **bytes** (converted from `limit_gb`).

* **`process: psutil.Process`**
  Represents the current process being monitored.

---

### Methods

#### `__init__(limit_gb: float = 4.0)`

Initialize the memory monitor with a configurable memory limit.

* **Parameters:**

  * `limit_gb` *(float, optional)* → Maximum memory allowed (in GB). Default = `4.0`.

* **Behavior:**

  * Converts the GB value into bytes.
  * Binds monitoring to the current running process (`psutil.Process()`).

---

#### `get_memory_usage() -> float`

Returns the **current memory usage** of the process.

* **Returns:**

  * *(float)* Memory usage in GB.

* **Example:**

  ```python
  monitor = MemoryMonitor(limit_gb=2)
  print(monitor.get_memory_usage())  # e.g., 0.25 (GB)
  ```

---

#### `check_memory_limit() -> bool`

Checks if the current memory usage is **below the configured limit**.

* **Returns:**

  * *(bool)* `True` if usage < limit, `False` otherwise.

* **Example:**

  ```python
  monitor = MemoryMonitor(limit_gb=1)
  if not monitor.check_memory_limit():
      print("Warning: Memory limit exceeded!")
  ```

---

#### `memory_guard()`

Context manager that **monitors memory usage during a block of code**.

* **Behavior:**

  * Records memory usage at block entry.
  * At block exit:

    * Logs **warning** if usage > limit.
    * Logs **info** if usage increased but within limit.

* **Usage Example:**

  ```python
  from memory_monitor import MemoryMonitor
  import logging

  logging.basicConfig(level=logging.INFO)

  monitor = MemoryMonitor(limit_gb=2)

  with monitor.memory_guard():
      big_data = [0] * (10**7)   # Example memory-heavy operation
  ```

* **Logs Example:**

  ```
  INFO:root:Memory increased during operation: 0.25 GB → 0.45 GB
  ```

  or

  ```
  WARNING:root:Memory usage exceeded limit: 2.10 GB (limit: 2.00 GB)
  ```

---

## Example Workflow

### 1. Monitoring Memory Growth

```python
monitor = MemoryMonitor(limit_gb=1)

with monitor.memory_guard():
    lst = [i for i in range(10**7)]
```

➡ Logs info about memory growth.

---

### 2. Enforcing Hard Limits

```python
monitor = MemoryMonitor(limit_gb=0.5)  # 512 MB

if not monitor.check_memory_limit():
    raise MemoryError("Process exceeded safe memory usage!")
```

➡ Can be extended to **raise exceptions** instead of just logging.

---

## Logging Behavior

* Uses Python's built-in `logging` module.
* Default logging level must be set by the user:

  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

| Event                            | Log Level | Message Example                                         |
| -------------------------------- | --------- | ------------------------------------------------------- |
| Memory increased but below limit | INFO      | `Memory increased during operation: 0.25 GB → 0.45 GB`  |
| Memory exceeded configured limit | WARNING   | `Memory usage exceeded limit: 2.10 GB (limit: 2.00 GB)` |

---

## Limitations

* Only monitors the **current Python process**, not system-wide usage.
* Logging only (does not terminate process automatically).
* Requires `psutil`, which may need installation on some systems.
* Granularity: Reports memory usage at context entry and exit, not continuously.

---

## Possible Extensions

* 🚨 Add option to **raise `MemoryError`** when exceeding limits.
* 📈 Periodic monitoring with background threads or async tasks.
* 🔔 Custom callbacks (e.g., send alert to monitoring system).
* 🎯 Integration with **system-level resource managers** (Kubernetes, Docker).
* 🧠 Track **peak memory usage** within context manager.

---

## Conclusion

The `MemoryMonitor` utility provides a **lightweight, effective tool** for monitoring and constraining memory usage in Python. Its **context manager design** makes it easy to drop into existing codebases without major refactoring.

It is especially useful for **RAG pipelines, ML serving, and ETL processes** where memory efficiency is critical.

---

```

Would you like me to **expand this further into ~400–600 lines** with **detailed examples, FAQs, and advanced usage scenarios** (like how to use it in FastAPI, ML training loops, etc.) — to match your `menemon`-style documentation?
```
