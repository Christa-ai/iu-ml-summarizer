"""
Unit tests for the summarizer, metrics and PDF-reader modules.

Each test is self-contained and imports its subject locally so that the
model is only loaded when a summarizer test actually runs (lazy import).
Run with:
    pytest tests/ -v
"""

import pytest


def test_summarize_returns_nonempty_string():
    """
    Verify that summarize() returns a non-empty string for a valid text input.

    This is a smoke test: it does not assert content quality, only that the
    model produces output without raising an exception.
    """
    from model.summarizer import summarize

    text = (
        "Scientists at CERN have discovered a new subatomic particle that could "
        "reshape our understanding of the standard model of physics. The particle, "
        "observed in high-energy proton collisions, displays properties that do not "
        "match any previously known particles. Researchers say further experiments "
        "are needed to confirm the finding and determine its exact characteristics."
    )
    result = summarize(text, max_len=60, min_len=10, num_beams=1)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_summarize_respects_max_length():
    """
    Verify that the generated summary does not exceed max_len tokens.

    A small buffer (+2) is added to account for the BOS/EOS special tokens
    that BART appends during tokenization of the decoded output.
    """
    from model.summarizer import summarize
    from transformers import BartTokenizer

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    text = "The quick brown fox jumps over the lazy dog. " * 20
    result = summarize(text, max_len=40, min_len=5, num_beams=1)
    tokens = tokenizer(result, return_tensors="pt")["input_ids"]
    assert tokens.shape[1] <= 42  # small buffer for special tokens


def test_compute_rouge_returns_valid_scores():
    """
    Verify that compute_rouge() returns ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    in the valid range [0.0, 1.0] for a simple prediction/reference pair.
    """
    from utils.metrics import compute_rouge

    prediction = "The cat sat on the mat."
    reference = "A cat was sitting on a mat."
    scores = compute_rouge(prediction, reference)

    assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


def test_average_rouge_correct():
    """
    Verify that average_rouge() computes the arithmetic mean for each metric
    across a list of per-article score dicts.
    """
    from utils.metrics import average_rouge

    score_list = [
        {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3},
        {"rouge1": 0.6, "rouge2": 0.4, "rougeL": 0.5},
    ]
    avg = average_rouge(score_list)
    assert avg["rouge1"] == pytest.approx(0.5, abs=1e-4)
    assert avg["rouge2"] == pytest.approx(0.3, abs=1e-4)
    assert avg["rougeL"] == pytest.approx(0.4, abs=1e-4)


def test_pdf_reader_invalid_input_raises():
    """
    Verify that extract_text_from_b64() raises an exception for malformed
    input that cannot be decoded as a valid base64-encoded PDF.
    """
    from utils.pdf_reader import extract_text_from_b64

    with pytest.raises(Exception):
        extract_text_from_b64("not-valid-base64-content")
