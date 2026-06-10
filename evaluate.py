"""
Evaluation pipeline for the BART summarizer.

Usage:
    python evaluate.py --samples 50 --output data/results/eval_results.json
    python evaluate.py --samples 50 --batch-size 8 --output data/results/eval_results.json
"""

import argparse
import json
import logging
import os
from datetime import datetime

# Use cached data — avoids network timeouts on Windows
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset

from model.summarizer import summarize_batch
from utils.metrics import Timer, average_rouge, compute_rouge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_evaluation(num_samples: int, output_path: str, batch_size: int = 8) -> None:
    log.info("Loading cnn_dailymail (split=test) ...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

    samples = dataset.select(range(min(num_samples, len(dataset))))
    log.info("Evaluating %d articles (batch_size=%d) ...", len(samples), batch_size)

    results = []
    total_timer = Timer()

    with total_timer:
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch = samples.select(range(batch_start, batch_end))

            articles   = [item["article"]    for item in batch]
            references = [item["highlights"] for item in batch]

            with Timer() as t:
                predictions = summarize_batch(articles)

            per_article_time = round(t.elapsed / len(articles), 3)

            for j, (pred, ref) in enumerate(zip(predictions, references)):
                scores = compute_rouge(pred, ref)
                scores["inference_s"] = per_article_time
                results.append(scores)

                i = batch_start + j
                if (i + 1) % 10 == 0:
                    log.info(
                        "  [%d/%d]  R1=%.4f  R2=%.4f  RL=%.4f  time=%.2fs/article",
                        i + 1, len(samples),
                        scores["rouge1"], scores["rouge2"], scores["rougeL"],
                        per_article_time,
                    )

    avg = average_rouge(results)
    avg_time = round(sum(r["inference_s"] for r in results) / len(results), 3)

    summary = {
        "model": "facebook/bart-large-cnn",
        "dataset": "cnn_dailymail 3.0.0",
        "num_samples": len(results),
        "batch_size": batch_size,
        "total_time_s": round(total_timer.elapsed, 1),
        "avg_inference_s": avg_time,
        "avg_rouge1": avg["rouge1"],
        "avg_rouge2": avg["rouge2"],
        "avg_rougeL": avg["rougeL"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    log.info("\n=== Results ===")
    for k, v in summary.items():
        log.info("  %-22s %s", k, v)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {"summary": summary, "per_article": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BART summarizer on cnn_dailymail")
    parser.add_argument("--samples",    type=int, default=50,  help="Number of articles to evaluate")
    parser.add_argument("--batch-size", type=int, default=8,   help="Articles per batch")
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/eval_results.json",
        help="Path for JSON output",
    )
    args = parser.parse_args()
    run_evaluation(args.samples, args.output, args.batch_size)
