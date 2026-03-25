FROM python:3.11-slim

WORKDIR /app

# No system packages needed for opencv-python-headless!

# Copy requirements
COPY pyproject.toml .

# Install dependencies (headless opencv needs nothing special)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    opencv-python-headless>=4.9.0.80 \
    -e .

# Copy app
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
