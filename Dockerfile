FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opencv-python-headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    opencv-python-headless>=4.9.0.80 \
    -e .

# Copy app code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
