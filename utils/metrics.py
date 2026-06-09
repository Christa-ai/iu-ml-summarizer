import time
from rouge_score import rouge_scorer


_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(prediction: str, reference: str) -> dict:
    """Return ROUGE-1, ROUGE-2 and ROUGE-L F1 scores for a single pair."""
    scores = _scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def average_rouge(score_list: list[dict]) -> dict:
    """Compute mean ROUGE scores across a list of result dicts."""
    keys = ["rouge1", "rouge2", "rougeL"]
    return {
        k: round(sum(s[k] for s in score_list) / len(score_list), 4)
        for k in keys
    }


class Timer:
    """Context manager that measures wall-clock time in seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = round(time.perf_counter() - self._start, 3)
