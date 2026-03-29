FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/
COPY train.py play.py main.py ./

# Install dependencies (agent extras include torch + tensorboard)
RUN uv sync --extra agent --no-dev

# Checkpoints and logs live on a mounted volume, not in the container
VOLUME /data
ENV CHECKPOINT_DIR=/data/checkpoints
ENV LOG_DIR=/data/runs

# TensorBoard port
EXPOSE 6006

# Default: print help
CMD ["uv", "run", "python", "train.py", "--help"]
