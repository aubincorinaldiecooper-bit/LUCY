# Minimal image — fine for dummy mode / CPU smoke tests.
# For the real model, switch the base to a CUDA image (e.g.
# nvidia/cuda:12.1.1-runtime-ubuntu22.04 + python) and install torch/omnivoice.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py engine.py ./

# Defaults; override in your deploy. Set OMNIVOICE_DUMMY=true to run without a model.
ENV OMNIVOICE_DEVICE=cuda \
    OMNIVOICE_DUMMY=false

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
