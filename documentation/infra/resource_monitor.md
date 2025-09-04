# Resource Monitor (`resource_monitor.py`)

---

## Overview

The **Resource Monitor** module provides a utility class, `ResourceMonitor`, for **tracking system resource usage** including CPU, memory, disk, and GPU.  

It is designed for use in **server monitoring, observability dashboards, or long-running applications** where system stability is critical.  

Key features include:  
- Real-time monitoring of CPU, memory, disk, and GPU utilization  
- Alerting via logging when resource usage exceeds thresholds  
- Configurable intervals and thresholds via a central **Settings** module  
- Modular design for easy integration into larger systems  

---

## Dependencies

- **psutil** → Retrieves CPU, memory, and disk statistics  
- **pynvml** (optional) → Retrieves NVIDIA GPU usage  
- **src.helpers.Settings** → Centralized application configuration  
- **src.infra.setup_logging** → Sets up structured and colorful logging  

Install required dependencies:
```bash
pip install psutil pynvml
````

---

## Class: `ResourceMonitor`

### Purpose

The `ResourceMonitor` class continuously monitors system resources and logs warnings when predefined thresholds are exceeded.

---

### Attributes

| Attribute          | Type     | Description                                   |
| ------------------ | -------- | --------------------------------------------- |
| `cpu_threshold`    | float    | Maximum CPU usage (0.0–1.0) before warning    |
| `memory_threshold` | float    | Maximum memory usage (0.0–1.0) before warning |
| `disk_threshold`   | float    | Maximum disk usage (0.0–1.0) before warning   |
| `gpu_threshold`    | float    | Maximum GPU usage (0.0–1.0) before warning    |
| `gpu_available`    | bool     | Indicates if GPU monitoring is enabled        |
| `app_settings`     | Settings | Loaded application settings instance          |

---

### Methods

#### `__init__()`

Initializes thresholds from the application's `Settings`.

* Automatically detects whether GPU monitoring is enabled.

#### `check_cpu_usage() -> dict`

Checks the current CPU usage and logs:

* INFO for normal usage
* WARNING if usage exceeds `cpu_threshold`

**Returns:**

```python
{"cpu_usage": float}  # Usage as fraction (0.0–1.0)
{"error": str}         # If psutil fails
```

#### `check_memory_usage() -> dict`

Checks system memory usage and logs accordingly.

**Returns:**

```python
{"memory_usage": float}  # Fractional memory usage
{"error": str}           # If psutil fails
```

#### `check_disk_usage() -> dict`

Checks root partition disk usage.

**Returns:**

```python
{"disk_usage": float}  # Fractional disk usage
{"error": str}         # If psutil fails
```

#### `check_gpu_usage() -> dict`

Checks NVIDIA GPU utilization using `pynvml` if available.

**Returns:**

```python
{"gpu_usage": float}   # Fractional GPU usage
{"error": str}         # If GPU check fails or pynvml missing
```

---

### `start_monitoring() -> None`

Continuously monitors system resources at the interval defined in settings (`MONITOR_HEALTH_CHECK_INTERVAL_SEC`).

* Calls all individual resource check methods in a loop
* Logs all usage statistics and warnings
* Can be stopped gracefully with `Ctrl+C`

**Example:**

```python
monitor = ResourceMonitor()
monitor.start_monitoring()
```

---

## Settings Parameters

| Parameter                           | Type  | Description                                |
| ----------------------------------- | ----- | ------------------------------------------ |
| `MONITOR_CPU_THRESHOLD`             | float | CPU usage alert threshold (0.0–1.0)        |
| `MONITOR_MEMORY_THRESHOLD`          | float | Memory usage alert threshold (0.0–1.0)     |
| `MONITOR_DISK_THRESHOLD`            | float | Disk usage alert threshold (0.0–1.0)       |
| `MONITOR_GPU_THRESHOLD`             | float | GPU usage alert threshold (0.0–1.0)        |
| `GPU_AVAILABLE`                     | bool  | Enable or disable GPU monitoring           |
| `MONITOR_HEALTH_CHECK_INTERVAL_SEC` | int   | Interval between resource checks (seconds) |

---

## Usage Example

```python
from resource_monitor import ResourceMonitor

# Initialize monitor
monitor = ResourceMonitor()

# Start continuous monitoring
monitor.start_monitoring()
```

**Sample Log Output:**

```
2025-07-21 18:25:43,102 - RESOURCE-MONITOR-CORE - INFO - CPU Usage: 15.37%
2025-07-21 18:25:43,104 - RESOURCE-MONITOR-CORE - WARNING - High memory usage detected: 92.15%
2025-07-21 18:25:43,106 - RESOURCE-MONITOR-CORE - INFO - Disk Usage: 55.21%
```

---

## Notes

* GPU monitoring requires `pynvml` and a supported NVIDIA GPU.
* Designed to run standalone or integrated into **observability dashboards**.
* Graceful shutdown supported with `Ctrl+C`.
* Modular design allows extension for additional resource checks or alerting mechanisms.

---

## Limitations

* Only monitors resources on the local machine
* GPU monitoring is optional and limited to NVIDIA GPUs
* Alerts are currently logged; they do not automatically terminate processes
* Continuous monitoring may introduce slight overhead due to `psutil` calls

---

## Possible Extensions

* Add **email or Slack notifications** for threshold breaches
* Integrate with **Prometheus/Grafana** for metrics visualization
* Enable **historical logging** for trends over time
* Support multiple GPUs and GPU selection

---

## Conclusion

The `ResourceMonitor` module provides a **robust, real-time monitoring solution** for Python applications.
It enables developers to **track CPU, memory, disk, and GPU usage**, log alerts, and integrate system observability into production workflows.

```