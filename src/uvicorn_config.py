# uvicorn_config.py
# Configuration for running FastAPI with optimal concurrency

import multiprocessing
import uvicorn

# Calculate optimal worker count (CPU cores * 2 + 1, but cap at reasonable limit)
cpu_count = multiprocessing.cpu_count()
worker_count = min(cpu_count * 2 + 1, 8)  # Cap at 8 workers

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False
    )
