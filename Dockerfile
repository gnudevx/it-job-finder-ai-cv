FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps trước (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download và cache model lúc build — không download lại khi restart container
# Model ~1GB, chỉ download 1 lần duy nhất
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

COPY . .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# 1 worker vì model load vào RAM — nhiều worker sẽ load nhiều lần
CMD ["uvicorn", "app.main:app","--host","0.0.0.0","--port","8002","--workers","1"]
