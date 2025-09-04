# MongoDB Engine Connector Module (`mongo_engine.py`)

---

## Overview

The **MongoDB Engine Connector** module provides functionality to **connect, manage, and interact with a local MongoDB instance**.  

It is designed to simplify **database access** for local development, RAG pipelines, or ML systems, and optionally **launch MongoDB + Mongo Express via Docker**.

Key features include:

- Local MongoDB connection via `pymongo`  
- Optional automatic launching of MongoDB and Mongo Express containers using Docker  
- Web UI access via Mongo Express on port 8081  
- Logging of database connection status, configuration, and errors  

This module enables developers to quickly set up a **self-contained MongoDB environment** with minimal manual intervention.

---

## Dependencies

- `pymongo` → MongoDB client  
- `docker` → Optional for launching local MongoDB and Mongo Express  
- `logging` → Operational transparency  
- `src.infra.setup_logging` → Standardized logging  
- `src.helpers.get_settings` → Application configuration  

---

## Internal Functions

### `_launch_mongodb_docker() -> None`

Launches **MongoDB** and **Mongo Express** containers if they are not already running.

- **Behavior:**  
  - Checks if Docker is installed  
  - Creates and starts `mongodb` container on port 27017 if not running  
  - Creates and starts `mongo-express` container on port 8081 if not running  
  - Waits for containers to initialize before proceeding  
  - Logs status at each step  

- **Error Handling:**  
  - Logs Docker command failures  
  - Detects missing Docker installation  
  - Catches unexpected errors during container management  

---

### `_open_mongo_web_ui()`

Opens Mongo Express Web UI in the default browser.

- **Behavior:**  
  - Logs the URL of the web interface (`http://localhost:8081`)  
  - Catches exceptions if browser opening fails  

---

## Main Function

### `get_mongo_client() -> Optional[MongoClient]`

Creates and returns a **MongoDB client** connected to the local database.

- **Behavior:**  
  1. Launches Docker containers and Mongo Express UI if `MONGODB_ENABLE_WEB_UI` is enabled  
  2. Connects to the MongoDB URI from `MONGODB_LOCAL_URI` or defaults to `mongodb://localhost:27017`  
  3. Tests the connection by listing available databases  
  4. Logs connection success and database names  

- **Returns:**  
  - `MongoClient` instance if connection succeeds  
  - `None` if connection fails  

- **Error Handling:**  
  - `ConnectionFailure` → Logs connection failures  
  - `ConfigurationError` → Raises for misconfigured MongoDB  
  - Other exceptions → Logs unexpected errors  

---

## Usage Example

```python
from src.infra.mongo_engine import get_mongo_client

# Get MongoDB client
client = get_mongo_client()

if client:
    print("Connection to MongoDB succeeded.")
    print("Databases:", client.list_database_names())
else:
    print("Connection to MongoDB failed.")
````

---

## Notes

* **Web UI:**

  * Mongo Express allows browsing MongoDB in a web browser at `http://localhost:8081`
  * Automatically launched if `MONGODB_ENABLE_WEB_UI` is `True`

* **Docker Requirement:**

  * Docker must be installed and available in PATH to enable automatic container launch

* **Local Development:**

  * Default MongoDB port: `27017`
  * Default Mongo Express port: `8081`

---

## Best Practices

1. **Check Docker Installation:** Ensure Docker is installed to use web UI features.
2. **Enable Web UI for Debugging:** `MONGODB_ENABLE_WEB_UI=True` allows easy database inspection.
3. **Connection Testing:** Always verify `get_mongo_client()` returns a valid client before database operations.
4. **Container Reuse:** Existing MongoDB and Mongo Express containers are reused if already created.

---

## Common Errors

| Error                                     | Cause                                               | Resolution                                          |
| ----------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| `MongoDB connection failed`               | MongoDB not running or misconfigured                | Start MongoDB manually or check `MONGODB_LOCAL_URI` |
| `MongoDB configuration error`             | Invalid MongoDB client settings                     | Correct URI or client configuration                 |
| `Docker is not installed or not in PATH`  | Web UI or container launch attempted without Docker | Install Docker and retry                            |
| `Unexpected error while launching Docker` | Unexpected runtime issue                            | Inspect logs and fix container setup                |

---

## Conclusion

The `mongo_engine.py` module provides a **reliable and convenient connector** to local MongoDB instances, with optional **automatic container launch and web UI access**.

It is ideal for **development environments, RAG pipelines, and ML workflows** requiring fast access to MongoDB without manual setup.

```
