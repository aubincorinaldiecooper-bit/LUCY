FROM python:3.11-slim

WORKDIR /app

ENV UV_CACHE_DIR=/app/cache/uv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN mkdir -p /app/cache/uv

COPY pyproject.toml uv.lock ./
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv sync --frozen --no-dev
RUN uv cache clean 2>/dev/null; rm -rf /root/.cache /tmp/* /var/cache/apt/archives/*.deb
RUN python -c "from pipecat.services.kokoro.tts import KokoroTTSService; import torch; print('✅ Build imports OK')" || exit 1

ARG CACHE_BUST=3
COPY . .

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
