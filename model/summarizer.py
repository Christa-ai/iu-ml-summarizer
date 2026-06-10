"""
model/summarizer.py — BART Summarisation Model

Loads facebook/bart-large-cnn from the local Hugging Face cache and exposes
two public functions:

    summarize()       – single-document inference
    summarize_batch() – multi-document inference in one forward pass (faster on CPU)

The module sets HF_HUB_OFFLINE=1 at import time to prevent any network
requests to the Hugging Face Hub during inference (required for air-gapped
on-premise deployments).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import BartTokenizer, BartForConditionalGeneration

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)


def summarize(
    text: str,
    max_len: int = 130,
    min_len: int = 30,
    num_beams: int = 1,
) -> str:
    """
    Generate an abstractive summary for a single document.

    Parameters
    ----------
    text      : Input document (plain text). Texts longer than 1 024 tokens
                are automatically truncated by the tokenizer.
    max_len   : Maximum number of tokens in the generated summary.
    min_len   : Minimum number of tokens in the generated summary.
    num_beams : Beam width for beam search. Higher values improve quality
                at the cost of inference time.

    Returns
    -------
    str : Decoded summary string.
    """
    inputs = tokenizer(
        [text],
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=num_beams,
        max_length=max_len,
        min_length=min_len,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_batch(
    texts: list,
    max_len: int = 130,
    min_len: int = 30,
    num_beams: int = 1,
) -> list:
    """Process multiple articles in a single forward pass (faster on CPU)."""
    inputs = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=num_beams,
        max_length=max_len,
        min_length=min_len,
        no_repeat_ngram_size=3,
    )
    return [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]
