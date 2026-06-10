# IU ML Systems — Automatischer Dokumentzusammenfasser

An interactive web application that generates abstractive summaries of PDF
documents using **facebook/bart-large-cnn** (Hugging Face Transformers).
Built as a portfolio project for the IU course *ML Systems (DLMDSPMLSD01_D)*.

---

## Features

| Feature | Details |
|---|---|
| PDF Upload | Drag-and-drop or click-to-browse; up to 10 MB |
| BART Summarisation | Configurable max/min length and beam width |
| ROUGE Evaluation | Offline benchmark results embedded as Plotly charts |
| Structured Logging | Human-readable `app.log` + machine-readable `requests.jsonl` |
| Docker Deployment | Single-command `docker compose up` with Nginx reverse proxy and HTTPS |
| CI/CD | GitHub Actions: pytest + Docker smoke test on every push |

---

## Project Structure

```
.
├── app.py                   # Dash web application (entry point)
├── evaluate.py              # Offline ROUGE/inference-time benchmark
├── generate_report.py       # Renders data/results/eval_report.md
├── model/
│   └── summarizer.py        # BART model loading and inference
├── utils/
│   ├── metrics.py           # ROUGE scoring helpers + Timer
│   ├── pdf_reader.py        # Base64 PDF decoding via pypdf
│   └── logger.py            # Centralised logging (app.log + requests.jsonl)
├── tests/
│   └── test_summarizer.py   # pytest unit tests
├── data/
│   ├── results/             # eval_results.json + eval_report.md (gitignored logs)
│   └── logs/                # Runtime logs (gitignored)
├── docker/
│   ├── nginx.conf           # Nginx reverse-proxy config (HTTP → HTTPS)
│   └── generate-cert.sh     # Helper: creates self-signed TLS certificate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml # GitHub Actions CI pipeline
```

---

## Quick Start (Local)

### 1. Create virtual environment and install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the BART model (first run only)

```bash
python -c "from transformers import BartTokenizer, BartForConditionalGeneration; \
    BartTokenizer.from_pretrained('facebook/bart-large-cnn'); \
    BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')"
```

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

---

## Docker Deployment (On-Premise)

### 1. Generate self-signed TLS certificate

```bash
bash docker/generate-cert.sh
```

### 2. Build and start containers

```bash
docker compose up --build
```

The app is then available at `https://localhost` (port 443).  
HTTP traffic on port 80 is automatically redirected to HTTPS.

> **Note:** Browsers will show a certificate warning because the certificate is
> self-signed. For production use, replace `docker/certs/` with a valid
> certificate from a CA (e.g. Let's Encrypt).

---

## Running the Offline Evaluation

```bash
python evaluate.py         # runs ROUGE benchmark, writes data/results/eval_results.json
python generate_report.py  # creates data/results/eval_report.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Logging

Every summarisation request is logged to two locations:

| File | Format | Purpose |
|---|---|---|
| `data/logs/app.log` | Plain text | Human-readable debugging |
| `data/logs/requests.jsonl` | JSON Lines | Machine-readable; one record per request |

Each JSONL record contains: timestamp, filename, character count, page count,
inference parameters, summary, inference time (seconds), truncation flag, and
status.

---

## Security Notes

- PDFs are processed in memory only; no user data is written to disk beyond the
  opt-in log files.
- Sensitive configuration (model paths, secrets) should be provided via
  environment variables (see `docker-compose.yml`).
- HTTPS is enforced in the Docker deployment via Nginx.

---

## License

This project is for academic purposes only.
