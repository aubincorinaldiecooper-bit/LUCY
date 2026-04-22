FROM python:3.11-slim
WORKDIR /app
ENV UV_CACHE_DIR=/app/cache/uv
# 1. Install system dependencies required for OpenCV and WebRTC
# Even 'headless' OpenCV needs these libs in a slim Linux environment
RUN apt-get update && apt-get install -y \
 libxcb1 \
 libxrender1 \
 libxext6 \
 libsm6 \
 libice6 \
 libgl1 \
 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN mkdir -p /app/cache/uv
# 2. Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 3. Copy application source after dependency install so code changes
# invalidate only this layer (no manual cache busting required).
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

