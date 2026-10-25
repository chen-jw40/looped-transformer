from __future__ import annotations

import re
import sys
from decimal import Decimal, InvalidOperation

from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from ouro_pipeline.sari import corpus_sari


def _tokens(text: str):
    return [token for token in re.split(r"\s|\.", text) if token]


def _extract_option(text: str, choices: str = "ABCDE") -> str:
    """Extract an explicit multiple-choice answer without using the reference."""
    text = (text or "").strip()
    patterns = (
        rf"^(?:assistant\s*)?([{choices}])(?:\s*[.、:：)]|\s|$)",
        rf"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(?:answer\s*[:：]\s*)?([{choices}])(?:\s*[.、:：)]|\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return ""


def _extract_number(text: str) -> str:
    """Extract the model's numeric answer from its response section."""
    text = (text or "").split("\n\nQuestion:", 1)[0]
    number = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    explicit = re.findall(rf"(?i)(?:final\s+answer|answer|result|is|=)\s*(?:is|:|=)?\s*({number})\s*%?", text)
    candidates = explicit or re.findall(number, text)
    if not candidates:
        return ""
    value = candidates[-1].replace(",", "")
    try:
        normalized = Decimal(value).normalize()
    except InvalidOperation:
        return value
    return format(normalized, "f")


def _normalized_number(text: str) -> str:
    value = (text or "").strip().replace(",", "").rstrip("%")
    try:
        return format(Decimal(value).normalize(), "f")
    except InvalidOperation:
        return value


def _bleu(reference: str, hypothesis: str, gram: int):
    weights = tuple([1.0 / gram] * gram)
    return sentence_bleu([_tokens(reference)], _tokens(hypothesis), weights)


def _reasoning_metrics(predictions, references):
    pred_reasoning = [text[2:] if text else "" for text in predictions]
    ref_reasoning = [text[2:] if text else "" for text in references]
    bleu_1, bleu_4, rouge_l = [], [], []
    rouge = Rouge(metrics=["rouge-l"])
    for prediction, reference in zip(pred_reasoning, ref_reasoning):
        prediction = prediction.strip()
        reference = reference.strip()
        if not prediction or not reference:
            continue
        bleu_1.append(_bleu(reference, prediction, 1))
        bleu_4.append(_bleu(reference, prediction, 4))
        try:
            rouge_l.append(rouge.get_scores(reference, prediction, avg=True)["rouge-l"]["f"])
        except ValueError as exc:
            if "at least 1 sentence" not in str(exc):
                raise
            rouge_l.append(0.0)
    denominator = max(1, len(references))
    return {
        "bleu-1": sum(bleu_1) / denominator,
        "bleu-4": sum(bleu_4) / denominator,
        "rouge-L": sum(rouge_l) / denominator,
    }


def _text_metrics(predictions, references):
    bleu_1, bleu_4, rouge_l = [], [], []
    rouge = Rouge(metrics=["rouge-l"])
    for prediction, reference in zip(predictions, references):
        # The rouge package raises for strings that contain no usable
        # sentence, including whitespace/punctuation-only generations.
        prediction = (prediction or "").strip()
        reference = (reference or "").strip()
        if not prediction or not reference:
            continue
        # rouge==1.0.1 reconstructs the LCS recursively. Long summaries can
        # exceed Python's default depth even though the dynamic-programming
        # computation itself is valid.
        required_depth = 2 * (len(prediction.split()) + len(reference.split())) + 100
        if required_depth > sys.getrecursionlimit():
            sys.setrecursionlimit(min(max(required_depth, 10_000), 50_000))
        bleu_1.append(_bleu(reference, prediction, 1))
        bleu_4.append(_bleu(reference, prediction, 4))
        try:
            rouge_l.append(rouge.get_scores(reference, prediction, avg=True)["rouge-l"]["f"])
        except ValueError as exc:
            if "at least 1 sentence" not in str(exc):
                raise
            # Treat malformed/empty text as a zero score while allowing the
            # remainder of the evaluation to complete.
            rouge_l.append(0.0)
    denominator = max(1, len(references))
    return {
        "bleu-1": sum(bleu_1) / denominator,
        "bleu-4": sum(bleu_4) / denominator,
        "rouge-L": sum(rouge_l) / denominator,
    }


def _postprocess_py150(code: str):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    for kind, literal in re.findall(pattern, code):
        code = code.replace(f"<{kind}_LIT:{literal}>", literal)
    return code


def evaluate_task(
    task: str,
    predictions: list[str],
    references: list[str],
    sources: list[str] | None = None,
):
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if task == "ScienceQA":
        pred_answers = [_extract_option(text) for text in predictions]
        ref_answers = [_extract_option(text) for text in references]
        accuracy = sum(p == r and p != "" for p, r in zip(pred_answers, ref_answers)) / max(1, len(references))
        format_failures = sum(answer not in {"A", "B", "C", "D", "E"} for answer in pred_answers)
        extra_metrics = _reasoning_metrics(predictions, references)
    elif task in {"MeetingBank", "20Minuten"}:
        accuracy = None
        format_failures = sum(not text.strip() for text in predictions)
        extra_metrics = _text_metrics(predictions, references)
        if task == "20Minuten":
            if sources is None:
                raise ValueError("20Minuten SARI requires source prompts")
            extra_metrics["sari"] = corpus_sari(sources, predictions, references)
    elif task == "Py150":
        processed_predictions = [_postprocess_py150(text) for text in predictions]
        processed_references = [_postprocess_py150(text) for text in references]
        similarity = sum(
            fuzz.ratio(prediction, reference) if prediction and reference else 0
            for prediction, reference in zip(processed_predictions, processed_references)
        ) / max(1, len(references))
        accuracy = None
        format_failures = sum(not text.strip() for text in predictions)
        extra_metrics = {"similarity": similarity}
    elif task in {"FOMC", "C-STANCE"}:
        pred_answers = [_extract_option(text, "ABC") for text in predictions]
        ref_answers = [_extract_option(text, "ABC") for text in references]
        accuracy = sum(p == r and p != "" for p, r in zip(pred_answers, ref_answers)) / max(1, len(references))
        format_failures = sum(not answer for answer in pred_answers)
        extra_metrics = {}
    elif task in {"NumGLUE-cm", "NumGLUE-ds"}:
        pred_answers = [_extract_number(text) for text in predictions]
        ref_answers = [_normalized_number(text) for text in references]
        accuracy = sum(p == r and p != "" for p, r in zip(pred_answers, ref_answers)) / max(1, len(references))
        format_failures = sum(not answer for answer in pred_answers)
        extra_metrics = {}
    else:
        accuracy = sum(p.strip() == r.strip() and p.strip() != "" for p, r in zip(predictions, references)) / max(1, len(references))
        format_failures = sum(not text.strip() for text in predictions)
        extra_metrics = {}
    return {
        "accuracy": accuracy,
        "format_failures": format_failures,
        "samples": len(references),
        "protocol": {
            "ScienceQA": "TRACE first-character answer accuracy plus reasoning BLEU/ROUGE",
            "MeetingBank": "TRACE BLEU/ROUGE",
            "20Minuten": "TRACE BLEU/ROUGE/SARI (13a tokenization)",
            "Py150": "TRACE literal-normalized fuzzy similarity",
        }.get(task, "TRACE strict exact match"),
        **extra_metrics,
    }
