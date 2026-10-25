"""Validate task-boundary recovery without modifying saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.parameters import training_config


def resume_plan(args):
    root = args.output_dir
    if root.resolve().is_relative_to(Path("result").resolve()):
        raise ValueError("Training recovery must write outside result/")
    if args.method not in {"full", "projected", "osft", "olora", "ewc", "lora"}:
        raise ValueError("Recovery supports full, activation projection, and DRSP")
    summary = json.loads((root / "summary.json").read_text())
    expected = training_config(args)
    saved = summary["config"]
    differences = {key: (saved.get(key), expected.get(key))
                   for key in saved.keys() | expected.keys()
                   if saved.get(key) != expected.get(key)}
    if differences:
        raise ValueError(f"Recovery must preserve the original configuration: {differences}")
    stages, reports = summary["stages"], summary["training"]
    if len(stages) != len(reports) or len(stages) > len(args.tasks):
        raise ValueError("Saved training reports and completed stages are inconsistent")
    checkpoint = args.model
    for index, (stage, report) in enumerate(zip(stages, reports), 1):
        task = args.tasks[index - 1]
        if (stage["stage"], stage["task"], report["stage"], report["task"]) != (index, task, index, task):
            raise ValueError("Completed tasks must be a contiguous prefix of the original sequence")
        checkpoint = root / "checkpoints" / f"{index:02d}_after_{task}"
        for name in ("config.json", "tokenizer_config.json", "modeling_ouro.py", "configuration_ouro.py"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        model_index = checkpoint / "model.safetensors.index.json"
        if model_index.is_file():
            shards = set(json.loads(model_index.read_text())["weight_map"].values())
        else:
            shards = {"model.safetensors"}
        for name in shards:
            if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
                raise FileNotFoundError(checkpoint / name)
        if args.method == "projected":
            basis = root / "bases" / f"{index:02d}_after_{task}.pt"
            if not basis.is_file() or not basis.stat().st_size:
                raise FileNotFoundError(basis)
        if args.method in {"osft", "olora", "ewc", "lora"}:
            continuation = root / "continuation" / f"{index:02d}_after_{task}.pt"
            if not continuation.is_file():
                raise FileNotFoundError(continuation)
        if args.method in {"olora", "lora", "ewc"}:
            folder = "ewc" if args.method == "ewc" else "adapters"
            history = root / folder / f"{index:02d}_after_{task}.pt"
            if not history.is_file():
                raise FileNotFoundError(history)
    # A task interrupted during checkpoint export requires manual artifact repair.
    recorded = {f"{stage['stage']:02d}_after_{stage['task']}" for stage in stages}
    unrecorded = {p.name for p in (root / "checkpoints").glob("*_after_*")} - recorded
    if unrecorded:
        raise ValueError(f"Uncommitted checkpoint directories require inspection: {sorted(unrecorded)}")
    return summary, len(stages), str(checkpoint)
