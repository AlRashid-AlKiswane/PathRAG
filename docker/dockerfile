# ==============================
# Stage 1: Base Arch Linux
# ==============================
FROM archlinux:latest AS base

WORKDIR /app

# In your existing RUN command, add ollama installation
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        python \
        python-pip \
        tesseract \
        tesseract-data-eng \
        ffmpeg \
        jq \
        perl \
        unzip \
        zip \
        p7zip \
        wget \
        rsync \
        base-devel \
        git \
        curl \
        docker \
    && pacman -Scc --noconfirm && \
    # Install Ollama CLI
    curl -fsSL https://ollama.ai/install.sh | sh

# ==============================
# Stage 2: Python dependencies
# ==============================
FROM base AS python-deps

COPY requirements.txt .

# Create virtual environment with Python 3.12
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# ==============================
# Stage 3: Runtime
# ==============================
FROM base AS runtime

# Copy virtual environment from python-deps stage
COPY --from=python-deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Expose the correct ports (based on your logs showing 8001)
EXPOSE 8001 27017 8081

CMD ["python", "src/uvicorn_config.py"]
