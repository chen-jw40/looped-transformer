#!/usr/bin/env python3
"""Test one existing seed's final checkpoint without modifying its training run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


from scripts.parameters import (REPO_ROOT, TASKS, TEST_SOURCES, PRIMARY_METRICS,
                                FINAL_CHECKPOINT, TEST_LOOPS, TEST_MAX_LENGTH,
                                generation_limits, parse_args)
from scripts.runtime import run_in_container


def save(summary, output):
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output / "summary.json")


def prepare(seed, output_root, options=None):
    options = options or parse_args(["test-seeds", "--seed", str(seed)])
    source = (REPO_ROOT / (options.source_run or TEST_SOURCES[seed])).resolve()
    raw = (source / "summary.json").read_bytes()
    training = json.loads(raw)
    config = training["config"]
    if (config["seed"] != seed or config["tasks"] != TASKS or config["loops"] != TEST_LOOPS
            or config["eval_loops"] != TEST_LOOPS or config["max_length"] != TEST_MAX_LENGTH):
        raise ValueError(f"Unexpected source protocol: {source}")
    checkpoint = source / "checkpoints" / FINAL_CHECKPOINT
    index = checkpoint / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    for filename in ["config.json", "modeling_ouro.py", *shards]:
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    data_root = (REPO_ROOT / (options.data_root or config["data_root"])).resolve()
    model = str((REPO_ROOT / (options.model or config["model"])).resolve())
    token_caps = generation_limits(vars(options))
    if options.max_new_tokens is not None:
        token_caps = {task: options.max_new_tokens for task in TASKS}
    datasets = {}
    for task in TASKS:
        path = data_root / task / "test.json"
        data = path.read_bytes()
        datasets[task] = {"path": str(path), "samples": len(json.loads(data)),
                          "sha256": hashlib.sha256(data).hexdigest()}
    output = output_root.resolve() / f"seed{seed}"
    protocol = {"source_run": str(source), "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
                "split": "test", "checkpoint": FINAL_CHECKPOINT, "tasks": TASKS,
                "loops": options.loop_depth, "max_length": options.max_length, "temperature": options.temperature,
                "max_new_tokens_by_task": token_caps, "eval_limit": None,
                "datasets": datasets}
    # Keep the original default protocol resumable for already submitted jobs.
    if model != str((REPO_ROOT / config["model"]).resolve()):
        protocol["tokenizer_model"] = model
    if not options.truncate_long_prompts:
        protocol["truncate_long_prompts"] = False
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if summary.get("test_protocol") != protocol:
            raise ValueError(f"Existing output has a different test protocol: {output}")
        if (output / "checkpoints" / FINAL_CHECKPOINT).resolve() != checkpoint.resolve():
            raise ValueError("Existing checkpoint link does not match the source")
        return output, summary
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "checkpoints" / FINAL_CHECKPOINT).symlink_to(checkpoint, target_is_directory=True)
    summary = {
        "config": {**config, "output_dir": str(output), "data_root": str(data_root),
                   "model": model, "eval_loops": options.loop_depth,
                   "max_length": options.max_length, "temperature": options.temperature,
                   "eval_tasks": TASKS, "eval_split": "test", "eval_limit": None,
                   "max_new_tokens": None, "max_new_tokens_by_task": token_caps,
                   "truncate_long_prompts": options.truncate_long_prompts, "wandb_mode": "disabled"},
        "method": training["method"], "test_protocol": protocol,
        "training": training["training"], "evaluations": {},
        "stage_status": {FINAL_CHECKPOINT: {"evaluation": "pending"}},
    }
    save(summary, output)
    return output, summary


def finalize(output):
    summary = json.loads((output / "summary.json").read_text())
    results = summary["evaluations"][FINAL_CHECKPOINT]
    scores = {}
    for task in TASKS:
        result = results[task]
        expected = summary["test_protocol"]["datasets"][task]
        if result["samples"] != expected["samples"]:
            raise ValueError(f"{task}: incomplete test evaluation")
        # Verify the actual saved prompts/references, not just the split label.
        data = Path(expected["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"{task}: test data changed since preparation")
        records = json.loads(data)
        count = 0
        with (output / "predictions" / FINAL_CHECKPOINT / f"{task}.jsonl").open() as handle:
            for count, line in enumerate(handle, 1):
                row = json.loads(line)
                record = records[count - 1]
                if (row["index"] != count - 1 or row["prompt"] != record["prompt"]
                        or row["reference"] != record["answer"]):
                    raise ValueError(f"{task}: predictions do not match test.json at row {count}")
        if count != len(records):
            raise ValueError(f"{task}: missing prediction rows")
        metric = PRIMARY_METRICS[task]
        score = result[metric] * (1 if metric in {"similarity", "sari"} else 100)
        scores[task] = {"metric": metric, "score_100": score, "samples": result["samples"]}
    macro = sum(item["score_100"] for item in scores.values()) / len(scores)
    summary["test_scores"] = {"tasks": scores, "macro_score_100": macro,
                              "prediction_rows_verified_against_test_json": True}
    summary["stage_status"][FINAL_CHECKPOINT]["evaluation"] = "complete"
    save(summary, output)
    print(f"Test complete: seed={summary['config']['seed']} macro={macro:.4f} output={output}", flush=True)


def run(args):
    output, summary = prepare(args.seed, args.output_root, args)
    print(json.dumps(summary["test_protocol"], ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    existing = summary.get("evaluations", {}).get(FINAL_CHECKPOINT, {})
    missing = [task for task in TASKS if task not in existing]
    if missing:
        run_in_container("vllm", ["worker", "--run-dir", str(output),
                        "--checkpoint", FINAL_CHECKPOINT, "--loop-depth", str(summary["test_protocol"]["loops"]), "--tasks", *missing,
                        "--batch-size", str(args.batch_size),
                        "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    finalize(output)
