"""
MongoDB Database Engine Connector Module.

This module provides a function to create and manage a connection
to a local MongoDB database.

Features:
- Connects to a local MongoDB instance.
- Supports launching local MongoDB + Mongo Express via Docker (if enabled).
- Logs configuration, errors, and database/collection info.

Usage:
    >>> from src.infra.mongo_engine import get_mongo_client
    >>> client = get_mongo_client()
    >>> if client:
    ...     print(client.list_database_names())
"""

import logging
import os
import sys
import subprocess
import time
from typing import Optional

try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.error("Failed to set up main directory path: %s", e)
    sys.exit(1)

from pymongo import MongoClient, errors
from src.infra import setup_logging
from src.helpers import get_settings, Settings

logger = setup_logging(name="MONGO-ENGINE")
app_settings: Settings = get_settings()


def _launch_mongodb_docker() -> None:
    """
    Launches MongoDB and Mongo Express using Docker if not already running.
    """
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)

        # Check if MongoDB container exists (running or stopped)
        result_db = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=mongodb", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        if "mongodb" not in result_db.stdout.strip():
            logger.info("Starting MongoDB container on port 27017...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "mongodb",
                "-p", "27017:27017",
                "-v", "mongo_data:/data/db",
                "mongo:latest"
            ], check=True)
            time.sleep(5)
            logger.info("MongoDB started successfully.")
        else:
            # Container exists, check if it's running
            result_running = subprocess.run(
                ["docker", "ps", "--filter", "name=mongodb", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            if "mongodb" not in result_running.stdout.strip():
                logger.info("Starting existing MongoDB container...")
                subprocess.run(["docker", "start", "mongodb"], check=True)
                time.sleep(5)
                logger.info("MongoDB started successfully.")
            else:
                logger.info("MongoDB container is already running.")

        # Check if Mongo Express container exists (running or stopped)
        result_express = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=mongo-express", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        if "mongo-express" not in result_express.stdout.strip():
            logger.info("Starting Mongo Express Web UI on port 8081...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "mongo-express",
                "--link", "mongodb:mongo",
                "-p", "8081:8081",
                "-e", "ME_CONFIG_MONGODB_SERVER=mongodb",
                "mongo-express:latest"
            ], check=True)
            time.sleep(3)
            logger.info("Mongo Express started successfully.")
        else:
            # Container exists, check if it's running
            result_express_running = subprocess.run(
                ["docker", "ps", "--filter", "name=mongo-express", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            if "mongo-express" not in result_express_running.stdout.strip():
                logger.info("Starting existing Mongo Express container...")
                subprocess.run(["docker", "start", "mongo-express"], check=True)
                time.sleep(3)
                logger.info("Mongo Express started successfully.")
            else:
                logger.info("Mongo Express container is already running.")

    except subprocess.CalledProcessError as e:
        logger.error("Failed to run Docker: %s", e.stderr if e.stderr else str(e))
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH.")
    except Exception as e:
        logger.error("Unexpected error while launching Docker: %s", str(e))


def _open_mongo_web_ui():
    """Opens Mongo Express Web UI in browser."""
    try:
        url = "http://localhost:8081"
        logger.info("Opened Mongo Express Web UI at: %s", url)
    except Exception as e:
        logger.warning("Could not open browser: %s", e)


def get_mongo_client() -> Optional[MongoClient]:
    """
    Creates and returns a MongoDB client instance connected to local MongoDB.

    Returns:
        MongoClient: MongoDB client instance if connection is successful.
        None: If connection fails.
    """
    try:
        if app_settings.MONGODB_ENABLE_WEB_UI and app_settings.MONGODB_LOCAL_URI:
            _launch_mongodb_docker()
            _open_mongo_web_ui()

        mongo_uri = app_settings.MONGODB_LOCAL_URI or "mongodb://localhost:27017"
        client = MongoClient(mongo_uri)
        logger.info("Connected to local MongoDB at: %s", mongo_uri)

        # Test connection by listing database names
        db_names = client.list_database_names()
        logger.debug("Existing databases: %s", db_names)

        return client

    except errors.ConnectionFailure as ce:
        logger.error("MongoDB connection failed: %s", ce)
        return None
    except errors.ConfigurationError as cfe:
        logger.error("MongoDB configuration error: %s", cfe)
        raise
    except Exception as e:
        logger.error("Unexpected error while connecting to MongoDB: %s", e)
        return None


if __name__ == "__main__":
    client = get_mongo_client()
    if client:
        print("Connection to MongoDB succeeded.")
        print("Databases:", client.list_database_names())
    else:
        print("Connection to MongoDB failed.")
