# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch BEFORE anything that depends on it,
# so docling's resolver sees the constraint already satisfied
# and never touches the CUDA wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY application ./application

EXPOSE 8000
ENV TORCH_COMPILE_DISABLE=1
ENV TORCHDYNAMO_DISABLE=1

CMD ["uvicorn", "application.ingestion.main:app", "--host", "0.0.0.0", "--port", "8000"]