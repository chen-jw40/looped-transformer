"""Atomic progress writes and TRACE input truncation shared by evaluators."""
import json
import os
from pathlib import Path

def save_summary(summary: dict, root: Path) -> None:
    temporary = root / ".summary.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / "summary.json")


def encode_prompt(tokenizer, prompt: str, max_length: int, task: str) -> tuple[list[int], bool]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    available = max_length - 1
    truncated = len(ids) > available
    if truncated:
        ids = ids[-available:] if task == "Py150" else ids[:available]
    return [tokenizer.bos_token_id] + ids, truncated


