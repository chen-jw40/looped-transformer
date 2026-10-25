"""Dependency-light SARI compatible with Hugging Face Evaluate's metric.

The scoring equations follow the Apache-2.0 Hugging Face Evaluate SARI
implementation, itself adapted from Tensor2Tensor.  Tokenization uses
SacreBLEU's standard 13a tokenizer.
"""
from __future__ import annotations

from collections import Counter

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a


_TOKENIZER = Tokenizer13a()


def _ngrams(tokens, order):
    return [" ".join(tokens[index:index + order]) for index in range(len(tokens) - order + 1)]


def _safe_f1(precision, recall):
    return 0.0 if precision == 0.0 and recall == 0.0 else 2 * precision * recall / (precision + recall)


def _operation_scores(source_grams, candidate_grams, reference_grams):
    reference_count = len(reference_grams)
    all_reference = Counter(gram for grams in reference_grams for gram in grams)
    source = Counter(source_grams)
    candidate = Counter(candidate_grams)
    source_repeated = Counter({gram: count * reference_count for gram, count in source.items()})
    candidate_repeated = Counter({gram: count * reference_count for gram, count in candidate.items()})

    kept = source_repeated & candidate_repeated
    good_kept = kept & all_reference
    reference_kept = source_repeated & all_reference
    keep_precision = 1.0
    keep_recall = 1.0
    if kept:
        keep_precision = sum(good_kept[g] / kept[g] for g in good_kept) / len(kept)
    if reference_kept:
        keep_recall = sum(good_kept.values()) / sum(reference_kept.values())

    deleted = source_repeated - candidate_repeated
    good_deleted = deleted - all_reference
    delete_precision = 1.0
    if deleted:
        delete_precision = sum(good_deleted[g] / deleted[g] for g in good_deleted) / len(deleted)

    added = set(candidate) - set(source)
    reference_added = set(all_reference) - set(source)
    good_added = added & set(all_reference)
    add_precision = len(good_added) / len(added) if added else 1.0
    add_recall = len(good_added) / len(reference_added) if reference_added else 1.0
    return _safe_f1(keep_precision, keep_recall), delete_precision, _safe_f1(add_precision, add_recall)


def sentence_sari(source: str, prediction: str, references: list[str]) -> float:
    if not references:
        raise ValueError("SARI requires at least one reference")
    source_tokens = _TOKENIZER(source.lower()).split()
    prediction_tokens = _TOKENIZER(prediction.lower()).split()
    reference_tokens = [_TOKENIZER(reference.lower()).split() for reference in references]
    keep = delete = add = 0.0
    for order in range(1, 5):
        scores = _operation_scores(
            _ngrams(source_tokens, order),
            _ngrams(prediction_tokens, order),
            [_ngrams(tokens, order) for tokens in reference_tokens],
        )
        keep += scores[0]
        delete += scores[1]
        add += scores[2]
    return 100.0 * ((keep / 4 + delete / 4 + add / 4) / 3)


def corpus_sari(sources: list[str], predictions: list[str], references: list[str]) -> float:
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("SARI sources, predictions, and references must have equal length")
    if not predictions:
        return 0.0
    return sum(
        sentence_sari(source, prediction, [reference])
        for source, prediction, reference in zip(sources, predictions, references)
    ) / len(predictions)
