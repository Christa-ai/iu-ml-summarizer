# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim for broad torch/transformers compatibility
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the BART model at build time ──────────────────────────────────
# This avoids any network access at container start (required for air-gapped
# on-premise deployments). The model is cached in the default HF cache dir.
RUN python - <<'EOF'
from transformers import BartTokenizer, BartForConditionalGeneration
BartTokenizer.from_pretrained("facebook/bart-large-cnn")
BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
print("Model cached successfully.")
EOF

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# Ensure runtime directories exist inside the container
RUN mkdir -p data/logs data/results

# ── Runtime configuration ──────────────────────────────────────────────────────
EXPOSE 8050

# workers=1: model is loaded in memory; multi-worker would duplicate it
# timeout=120: allows for longer inference on slower hardware
CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
