"""
Centralised logging for the Dokumentzusammenfasser app.

Two outputs, one location (data/logs/):
  app.log        – human-readable, all events (INFO + WARNING + ERROR)
  requests.jsonl – one JSON object per summarisation request (for re-training)
"""

import json
import logging
import os
from datetime import datetime

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "logs"
)

_REQUESTS_FILE = os.path.join(_LOG_DIR, "requests.jsonl")


def setup_logger() -> logging.Logger:
    """
    Configure root logger once at app startup.
    Returns the named 'app' logger for use throughout the application.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)

    log_file = os.path.join(_LOG_DIR, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = logging.getLogger("app")
    logger.info("[startup] Logger initialised → %s", log_file)
    return logger


def log_request(
    logger: logging.Logger,
    *,
    filename: str,
    char_count: int,
    pages: int,
    max_len: int,
    min_len: int,
    num_beams: int,
    summary: str,
    inference_s: float,
    truncated: bool,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """
    Append one JSON line to requests.jsonl and write a summary to app.log.

    Parameters
    ----------
    truncated : True when the input exceeded the model's 1 024-token limit.
    status    : 'ok' | 'error'
    error     : exception message when status == 'error'
    """
    entry: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "char_count": char_count,
        "pages": pages,
        "max_len": max_len,
        "min_len": min_len,
        "num_beams": num_beams,
        "summary_word_count": len(summary.split()) if summary else 0,
        "inference_s": round(inference_s, 3),
        "truncated": truncated,
        "status": status,
    }
    if error:
        entry["error"] = error

    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_REQUESTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if status == "ok":
        logger.info(
            "[request] %s | %d Zeichen | %d Seiten | Inferenz: %.2f s%s",
            filename,
            char_count,
            pages,
            inference_s,
            " | ⚠ Eingabe gekürzt (>1024 Tokens)" if truncated else "",
        )
    else:
        logger.error("[request] %s | Fehler: %s", filename, error)
