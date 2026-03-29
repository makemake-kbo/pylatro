FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv (manages its own Python)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/
COPY train.py play.py main.py README.md ./

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
