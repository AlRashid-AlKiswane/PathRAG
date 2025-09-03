# PathRAG - Advanced Document Processing & Retrieval System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Graph Building Methods](#graph-building-methods)
- [Document Processing Modes](#document-processing-modes)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

PathRAG is a cutting-edge document processing and retrieval system that combines advanced machine learning techniques with graph-based algorithms to provide intelligent document analysis and question-answering capabilities. The system supports multiple graph construction methods, OCR processing, and scalable parallel processing.

### Key Capabilities

- **Multi-format Document Support**: Process PDF, TXT, and image-based documents
- **Advanced OCR**: Extract text from images and scanned documents using Tesseract
- **Graph-based Retrieval**: Multiple graph construction algorithms for optimal performance
- **Scalable Architecture**: Support for parallel processing and multiple workers
- **Web UI**: Interactive interface for graph visualization and document management
- **Personal Graphs**: Individual user graphs for personalized document processing

## ✨ Features

- 🚀 **High Performance**: Multi-worker support with parallel processing
- 📊 **Multiple Graph Algorithms**: 7 different graph construction methods
- 🔍 **Advanced OCR**: Tesseract integration for image text extraction
- 🌐 **Web Interface**: Built-in UI for easy interaction
- 🗄️ **MongoDB Integration**: Robust document storage with Docker support
- 🔧 **Flexible Configuration**: Comprehensive environment-based settings
- 📱 **Cross-Platform**: Support for Windows, Linux (Ubuntu, Arch), and macOS
- 🎯 **Personal Workspaces**: Individual user graphs and processing contexts

## 💻 System Requirements

### Supported Operating Systems
- **Windows** 10/11
- **Linux**: Ubuntu 20.04+, Arch Linux
- **macOS**: 10.15+

### Required Software
- **Python** 3.8 or higher
- **Docker** (latest version)
- **MongoDB** (via Docker or local installation)

## 🛠️ Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3 python3-pip tesseract-ocr tesseract-ocr-eng ffmpeg docker.io jq perl unzip zip p7zip-full wget rsync
```

#### Arch Linux
```bash
sudo pacman -S python python-pip tesseract tesseract-data-eng ffmpeg docker jq perl unzip zip p7zip wget rsync
# If using yay
yay -S additional-packages-if-needed
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python tesseract ffmpeg docker jq perl wget rsync
brew install --cask docker
```

#### Windows
1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
4. Install [FFmpeg](https://ffmpeg.org/download.html)
5. Add all tools to your system PATH

### Step 2: Clone the Repository
```bash
git clone https://github.com/AlRashid-AlKiswane/PathRAG.git
cd pathrag
```

### Step 3: Create Python Virtual Environment

#### Linux/macOS
```bash
python3 -m venv pathrag_env
source pathrag_env/bin/activate
```

#### Windows
```cmd
python -m venv pathrag_env
pathrag_env\Scripts\activate
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

PathRAG uses two environment files for configuration:

### 1. Main Configuration (`.env`)
Copy the provided configuration template and customize:

```bash
cp .env.template .env
```

Key settings to configure:
- `MONGODB_LOCAL_URI`: MongoDB connection string
- `OLLAMA_HOST`: Ollama server endpoint
- `BUILD_GRAPH_METHOD`: Choose your preferred graph method
- `UPLOAD_STORAGE_PATH`: Document storage location

### 2. Uvicorn Server Configuration (`.uvicorn.env`)
Create the Uvicorn configuration file:

```bash
# .uvicorn.env
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8001
UVICORN_WORKERS=4
UVICORN_RELOAD=false

UVICORN_LOG_LEVEL=info
UVICORN_PROXY_HEADERS=true
UVICORN_FORWARDED_ALLOW_IPS=*

UVICORN_LIMIT_CONCURRENCY=100
UVICORN_BACKLOG=2048
UVICORN_TIMEOUT_KEEP_ALIVE=5
UVICORN_LIMIT_MAX_REQUESTS=10000

UVICORN_ACCESS_LOG=true
UVICORN_USE_COLORS=true
UVICORN_SERVER_HEADER=true
UVICORN_DATE_HEADER=true
```

## 🚀 Usage

### Starting the Application
```bash
# Activate virtual environment
source pathrag_env/bin/activate  # Linux/macOS
# or
pathrag_env\Scripts\activate     # Windows

# Run the application
python src/uvicorn_config.py
```

### Accessing the Interface
- **Web UI**: http://0.0.0.0:8000/
- **API Documentation**: http://0.0.0.0:8000/docs

## 🔧 Graph Building Methods

PathRAG supports multiple graph construction algorithms:

| Method | Description | Best For |
|--------|-------------|----------|
| `knn` | K-Nearest Neighbors | General purpose, balanced performance |
| `hierarchical` | Hierarchical clustering | Structured documents |
| `approximate` | Approximate nearest neighbors | Large datasets, speed priority |
| `multi_level` | Multi-level approach | Complex document relationships |
| `hybrid` | Combination of methods | Maximum accuracy |
| `lsh` | Locality Sensitive Hashing | High-dimensional data |
| `spectral` | Spectral clustering | Community detection |

Configure in `.env`:
```bash
BUILD_GRAPH_METHOD="knn"  # Change to your preferred method
```

## 📄 Document Processing Modes

### OCR Processing Options

1. **`ocr_only`**: Process only images in PDFs with OCR
   - Best for: Scanned documents, image-heavy PDFs
   - Processing: Extracts text from images only

2. **`no_ocr`**: Convert documents to chunks without OCR
   - Best for: Text-based PDFs, plain text documents
   - Processing: Fast text extraction, no image processing

3. **`all`**: Complete processing with OCR and chunking
   - Best for: Mixed content documents
   - Processing: Full OCR + text chunking (recommended)

## 📚 API Documentation
## 📁 Project Structure

```
src
 ┣ __pycache__
 ┃ ┣ __init__.cpython-313.pyc
 ┃ ┣ dependencies.cpython-313.pyc
 ┃ ┗ main.cpython-313.pyc
 ┣ controllers
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ chunking_documents.cpython-313.pyc
 ┃ ┃ ┣ ex_images_pdf.cpython-313.pyc
 ┃ ┃ ┣ life_span.cpython-313.pyc
 ┃ ┃ ┣ md_files_chunking.cpython-313.pyc
 ┃ ┃ ┣ ocr.cpython-313.pyc
 ┃ ┃ ┣ text_operation.cpython-313.pyc
 ┃ ┃ ┗ unique_filename_generator.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ chunking_documents.py
 ┃ ┣ ex_images_pdf.py
 ┃ ┣ md_files_chunking.py
 ┃ ┣ ocr.py
 ┃ ┣ ocr_handon.py
 ┃ ┣ text_operation.py
 ┃ ┗ unique_filename_generator.py
 ┣ helpers
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┗ settings.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┗ settings.py
 ┣ infra
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ logger.cpython-313.pyc
 ┃ ┃ ┣ memory_monitor.cpython-313.pyc
 ┃ ┃ ┗ resource_monitor.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ logger.py
 ┃ ┣ memory_monitor.py
 ┃ ┗ resource_monitor.py
 ┣ llms_providers
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ embedding.cpython-313.pyc
 ┃ ┃ ┗ ollama_provider.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ embedding.py
 ┃ ┗ ollama_provider.py
 ┣ mongodb
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ graph_clear_collection.cpython-313.pyc
 ┃ ┃ ┣ graph_collection.cpython-313.pyc
 ┃ ┃ ┣ graph_engin.cpython-313.pyc
 ┃ ┃ ┣ graph_insert.cpython-313.pyc
 ┃ ┃ ┣ graph_pull_from_collection.cpython-313.pyc
 ┃ ┃ ┣ mongodb_clear_collection.cpython-313.pyc
 ┃ ┃ ┣ mongodb_collection.cpython-313.pyc
 ┃ ┃ ┣ mongodb_engin.cpython-313.pyc
 ┃ ┃ ┣ mongodb_insert.cpython-313.pyc
 ┃ ┃ ┗ mongodb_pull_from_collection.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ mongodb_clear_collection.py
 ┃ ┣ mongodb_collection.py
 ┃ ┣ mongodb_engin.py
 ┃ ┣ mongodb_insert.py
 ┃ ┗ mongodb_pull_from_collection.py
 ┣ prompt
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┗ prompt_templates.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┗ prompt_templates.py
 ┣ rag
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ graph_cache.cpython-313.pyc
 ┃ ┃ ┣ path_rag_factory.cpython-313.pyc
 ┃ ┃ ┣ path_rag_metrics.cpython-313.pyc
 ┃ ┃ ┣ pathrag.cpython-313.pyc
 ┃ ┃ ┗ plot_graph.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ graph_cache.py
 ┃ ┣ path_rag_factory.py
 ┃ ┣ path_rag_metrics.py
 ┃ ┣ pathrag.py
 ┃ ┗ plot_graph.py
 ┣ routes
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ build_path_rag.cpython-313.pyc
 ┃ ┃ ┣ chatbot.cpython-313.pyc
 ┃ ┃ ┣ chunking_docs.cpython-313.pyc
 ┃ ┃ ┣ embedding_chunks.cpython-313.pyc
 ┃ ┃ ┣ live_retrevel.cpython-313.pyc
 ┃ ┃ ┣ resource_monitor.cpython-313.pyc
 ┃ ┃ ┣ route_chunker_md_files.cpython-313.pyc
 ┃ ┃ ┣ storage_management.cpython-313.pyc
 ┃ ┃ ┣ upload_files.cpython-313.pyc
 ┃ ┃ ┗ user_file.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ build_path_rag.py
 ┃ ┣ chatbot.py
 ┃ ┣ chunking_docs.py
 ┃ ┣ embedding_chunks.py
 ┃ ┣ live_retrevel.py
 ┃ ┣ resource_monitor.py
 ┃ ┣ route_chunker_md_files.py
 ┃ ┣ storage_management.py
 ┃ ┣ upload_files.py
 ┃ ┗ user_file.py
 ┣ schemas
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ build_method_graph.cpython-313.pyc
 ┃ ┃ ┣ chatbot.cpython-313.pyc
 ┃ ┃ ┣ checkpoint_meta_data.cpython-313.pyc
 ┃ ┃ ┣ chunker_route.cpython-313.pyc
 ┃ ┃ ┣ md_chunks.cpython-313.pyc
 ┃ ┃ ┣ ocr_core.cpython-313.pyc
 ┃ ┃ ┗ rag.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ build_method_graph.py
 ┃ ┣ chatbot.py
 ┃ ┣ checkpoint_meta_data.py
 ┃ ┣ chunker_route.py
 ┃ ┣ md_chunks.py
 ┃ ┣ ocr_core.py
 ┃ ┗ rag.py
 ┣ utils
 ┃ ┣ __pycache__
 ┃ ┃ ┣ __init__.cpython-313.pyc
 ┃ ┃ ┣ auto_save_manager.cpython-313.pyc
 ┃ ┃ ┣ checkpoint.cpython-313.pyc
 ┃ ┃ ┣ clean_md_contect.cpython-313.pyc
 ┃ ┃ ┣ do_senitize.cpython-313.pyc
 ┃ ┃ ┣ ollama_maneger.cpython-313.pyc
 ┃ ┃ ┣ size_file.cpython-313.pyc
 ┃ ┃ ┣ thred_safe_path_rag.cpython-313.pyc
 ┃ ┃ ┗ timer_decorator.cpython-313.pyc
 ┃ ┣ __init__.py
 ┃ ┣ auto_save_manager.py
 ┃ ┣ checkpoint.py
 ┃ ┣ clean_md_contect.py
 ┃ ┣ do_senitize.py
 ┃ ┣ ollama_maneger.py
 ┃ ┣ size_file.py
 ┃ ┣ thred_safe_path_rag.py
 ┃ ┗ timer_decorator.py
 ┣ web
 ┃ ┗ index.html
 ┣ __init__.py
 ┣ dependencies.py
 ┣ main.py
 ┗ uvicorn_config.py
```

## 🔧 Troubleshooting

### Common Issues

#### MongoDB Connection Error
```bash
# Check if MongoDB container is running
docker ps | grep mongo

# Start MongoDB if not running
docker start pathrag-mongodb
```

#### Tesseract Not Found
```bash
# Verify Tesseract installation
tesseract --version

# Linux: Install if missing
sudo apt install tesseract-ocr tesseract-ocr-eng
```

#### Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep 8000

# Kill the process or change port in .uvicorn.env
```

#### Permission Issues (Linux)
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
chmod +x src/uvicorn_config.py
```

### Performance Optimization

#### For Low Memory Systems
```bash
# Use memory-constrained settings in .env
MEMORY_MAX_WORKERS=2
MEMORY_BATCH_SIZE=200
MEMORY_CACHE_SIZE_MB=50
```

#### For High Performance
```bash
# Increase workers and batch sizes
PROD_MAX_WORKERS=8
PROD_BATCH_SIZE=2000
PROD_CACHE_SIZE_MB=1000
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [MongoDB](https://www.mongodb.com/) for document storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Ollama](https://ollama.ai/) for language model integration


---

**Made with ❤️ by the PathRAG Team**