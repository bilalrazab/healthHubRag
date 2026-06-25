# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=false \
    CHROMA_TELEMETRY=false

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────
COPY . .

# ── Ensure data directories exist ────────────────────────────
RUN mkdir -p data/db data/chroma data/eval data/raw \
             data/structured data/simulated

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────
CMD ["uvicorn", "interfaces.api:app", "--host", "0.0.0.0", "--port", "8000"]
