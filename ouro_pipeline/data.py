from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


from scripts.parameters import TASKS


REPLAY_DATASETS = ("Lima",)


def read_split(data_root: str | Path, task: str, split: str) -> list[dict[str, str]]:
    path = Path(data_root) / task / f"{split}.json"
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: top-level JSON value must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("answer"), str):
            raise ValueError(f"{path}: item {index} must contain string prompt and answer fields")
    return records


def fixed_subset(records: list[dict[str, str]], limit: int | None, seed: int):
    if limit is None or limit >= len(records):
        return records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in sorted(indices[:limit])]


class PromptAnswerDataset(Dataset):
    def __init__(self, records: list[dict[str, str]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


@dataclass
class AnswerOnlyCollator:
    tokenizer: object
    max_length: int = 1024
    inference: bool = False
    prompt_truncation: str = "error"

    def _encode(self, record):
        prompt_ids = self.tokenizer.encode(record["prompt"], add_special_tokens=False)
        answer_ids = self.tokenizer.encode(record["answer"], add_special_tokens=False)
        eos = [self.tokenizer.eos_token_id]
        reserved = 1 if self.inference else 2 + len(answer_ids)
        available_prompt = self.max_length - reserved
        if available_prompt < 0:
            raise ValueError(
                f"Answer requires {reserved} tokens and cannot fit max_length={self.max_length}"
            )
        if len(prompt_ids) > available_prompt:
            if self.prompt_truncation == "left":
                prompt_ids = prompt_ids[-available_prompt:] if available_prompt else []
            elif self.prompt_truncation == "right":
                prompt_ids = prompt_ids[:available_prompt]
            else:
                raise ValueError(
                    f"Sample prompt requires {len(prompt_ids)} tokens but only {available_prompt} fit; "
                    "choose an explicit prompt truncation policy"
                )
        if self.inference:
            ids = [self.tokenizer.bos_token_id] + prompt_ids
            labels = None
        else:
            ids = [self.tokenizer.bos_token_id] + prompt_ids + answer_ids + eos
            labels = [-100] * (1 + len(prompt_ids)) + answer_ids + eos
        assert len(ids) <= self.max_length
        return ids, labels

    def __call__(self, records):
        encoded = [self._encode(record) for record in records]
        max_len = max(len(ids) for ids, _ in encoded)
        input_ids, attention_mask, labels = [], [], []
        for ids, item_labels in encoded:
            pad = max_len - len(ids)
            # Right padding for training, left padding for batched generation.
            if self.inference:
                input_ids.append([self.tokenizer.pad_token_id] * pad + ids)
                attention_mask.append([0] * pad + [1] * len(ids))
            else:
                input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
                attention_mask.append([1] * len(ids) + [0] * pad)
                labels.append(item_labels + [-100] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if not self.inference:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def token_length_report(records, tokenizer, max_length):
    lengths = []
    answer_lengths = []
    for record in records:
        prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(record["answer"], add_special_tokens=False)
        lengths.append(2 + len(prompt) + len(answer))
        answer_lengths.append(1 + len(answer))
    ordered = sorted(lengths)
    percentile_index = max(0, int(0.95 * len(ordered)) - 1) if ordered else 0
    return {
        "samples": len(lengths),
        "max_tokens": max(lengths, default=0),
        "p95_tokens": ordered[percentile_index] if ordered else 0,
        "max_answer_tokens": max(answer_lengths, default=0),
        "over_max_length": sum(length > max_length for length in lengths),
    }
