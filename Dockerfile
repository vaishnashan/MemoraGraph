# MemoraGraph — FastAPI backend
# FalkorDB and Supabase are cloud-hosted (see .env), so only the app
# itself needs containerizing.

FROM python:3.11-slim

WORKDIR /app

# System deps Docling/torch/transformers commonly need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt
COPY application ./application

EXPOSE 8000

CMD ["uvicorn", "application.main:app", "--host", "0.0.0.0", "--port", "8000"]
