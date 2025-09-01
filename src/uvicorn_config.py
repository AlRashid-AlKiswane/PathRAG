"""
Uvicorn Server Configuration and Launcher.

This module defines a configuration model for running a Uvicorn ASGI server.
It loads settings from environment variables or from a `.uvicorn.env` file
(if available), providing a convenient way to manage deployment and development
configurations consistently.

The main entry point (`if __name__ == "__main__"`) starts the server using
the validated configuration.
"""

import multiprocessing
import uvicorn
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_workers() -> int:
    """Return a safe default worker count based on CPU cores."""
    cpu = multiprocessing.cpu_count()
    return min(cpu * 2 + 1, 8)


def str_to_bool(val: str | None, default: bool = False) -> bool:
    """Convert string environment values to a boolean."""
    return val.lower() in ("1", "true", "yes", "on") if isinstance(val, str) else default


class UvicornConfig(BaseSettings):
    """
    Configuration model for Uvicorn server.

    Attributes can be overridden using environment variables or loaded from
    a `.uvicorn.env` file. Default values are provided for all settings to
    ensure the server can start without explicit configuration.

    Example:
        Create a `.uvicorn.env` file:

            UVICORN_HOST=127.0.0.1
            UVICORN_PORT=8080
            UVICORN_RELOAD=true

        Then run:

            python uvicorn_config.py
    """

    UVICORN_HOST: str = "0.0.0.0"
    UVICORN_PORT: int = 8000
    UVICORN_WORKERS: int = get_default_workers()
    UVICORN_RELOAD: bool = False
    UVICORN_LOG_LEVEL: str = "info"
    UVICORN_PROXY_HEADERS: bool = True
    UVICORN_FORWARDED_ALLOW_IPS: str = "*"
    UVICORN_LIMIT_CONCURRENCY: int = 100
    UVICORN_BACKLOG: int = 2048
    UVICORN_TIMEOUT_KEEP_ALIVE: int = 5
    UVICORN_LIMIT_MAX_REQUESTS: int = 10000
    UVICORN_ACCESS_LOG: bool = True
    UVICORN_USE_COLORS: bool = True
    UVICORN_SERVER_HEADER: bool = True
    UVICORN_DATE_HEADER: bool = True
    UVICORN_SSL_KEYFILE: str | None = None
    UVICORN_SSL_CERTFILE: str | None = None
    UVICORN_SSL_CA_CERTS: str | None = None
    UVICORN_SSL_KEYFILE_PASSWORD: str | None = None

    model_config = SettingsConfigDict(
        env_file=".uvicorn.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_config() -> UvicornConfig:
    """Load Uvicorn configuration from environment variables or `.uvicorn.env`."""
    return UvicornConfig()


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        app="main:app",
        host=cfg.UVICORN_HOST,
        port=cfg.UVICORN_PORT,
        workers= 1, # cfg.UVICORN_WORKERS,
        reload=cfg.UVICORN_RELOAD,
        log_level=cfg.UVICORN_LOG_LEVEL,
        proxy_headers=cfg.UVICORN_PROXY_HEADERS,
        forwarded_allow_ips=cfg.UVICORN_FORWARDED_ALLOW_IPS,
        limit_concurrency=cfg.UVICORN_LIMIT_CONCURRENCY,
        backlog=cfg.UVICORN_BACKLOG,
        timeout_keep_alive=cfg.UVICORN_TIMEOUT_KEEP_ALIVE,
        limit_max_requests=cfg.UVICORN_LIMIT_MAX_REQUESTS,
        access_log=cfg.UVICORN_ACCESS_LOG,
        use_colors=cfg.UVICORN_USE_COLORS,
        server_header=cfg.UVICORN_SERVER_HEADER,
        date_header=cfg.UVICORN_DATE_HEADER,
        ssl_keyfile=cfg.UVICORN_SSL_KEYFILE,
        ssl_certfile=cfg.UVICORN_SSL_CERTFILE,
        ssl_ca_certs=cfg.UVICORN_SSL_CA_CERTS,
        ssl_keyfile_password=cfg.UVICORN_SSL_KEYFILE_PASSWORD,
    )
