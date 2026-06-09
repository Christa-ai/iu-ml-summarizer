from transformers import BartTokenizer, BartForConditionalGeneration

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)


def summarize(
    text: str,
    max_len: int = 130,
    min_len: int = 30,
    num_beams: int = 4,
) -> str:
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
