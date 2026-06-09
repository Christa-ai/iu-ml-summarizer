"""
Evaluation pipeline for the BART summarizer.

Usage:
    python evaluate.py --samples 200 --output data/results/eval_results.json
"""

import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset

from model.summarizer import summarize
from utils.metrics import Timer, average_rouge, compute_rouge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_evaluation(num_samples: int, output_path: str) -> None:
    log.info("Loading cnn_dailymail (split=test) ...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

    samples = dataset.select(range(min(num_samples, len(dataset))))
    log.info("Evaluating %d articles ...", len(samples))

    results = []
    total_timer = Timer()

    with total_timer:
        for i, item in enumerate(samples):
            article = item["article"]
            reference = item["highlights"]

            with Timer() as t:
                prediction = summarize(article)

            scores = compute_rouge(prediction, reference)
            scores["inference_s"] = t.elapsed

            results.append(scores)

            if (i + 1) % 10 == 0:
                log.info(
                    "  [%d/%d]  R1=%.4f  R2=%.4f  RL=%.4f  time=%.2fs",
                    i + 1, len(samples),
                    scores["rouge1"], scores["rouge2"], scores["rougeL"],
                    t.elapsed,
                )

    avg = average_rouge(results)
    avg_time = round(sum(r["inference_s"] for r in results) / len(results), 3)

    summary = {
        "model": "facebook/bart-large-cnn",
        "dataset": "cnn_dailymail 3.0.0",
        "num_samples": len(results),
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
    parser.add_argument("--samples", type=int, default=200, help="Number of articles to evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/eval_results.json",
        help="Path for JSON output",
    )
    args = parser.parse_args()
    run_evaluation(args.samples, args.output)
