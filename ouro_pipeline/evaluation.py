"""Host-side checkpoint orchestration with split-isolated, resumable outputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ouro_pipeline.evaluation_io import save_summary
from scripts.parameters import REPO_ROOT, PRIMARY_METRICS
from scripts.runtime import run_in_container


def checkpoint_plan(summary, run_dir):
    return [(f"{stage:02d}_after_{task}", stage)
            for stage, task in enumerate(summary["config"]["tasks"], 1)
            if (run_dir / "checkpoints" / f"{stage:02d}_after_{task}").is_dir()]


def tasks_by_depth(summary, run_dir, checkpoint, stage):
    config = summary["config"]
    tasks = config["tasks"]
    requested = config.get("eval_tasks") or tasks
    selected = [task for task in requested if stage == 0 or task in tasks[:stage]]
    existing = summary.get("evaluations", {}).get(checkpoint, {})
    grouped = {}
    for task in selected:
        if task not in existing:
            depth = config.get("eval_loops", config["loops"])
            grouped.setdefault(depth, []).append(task)
    return grouped, selected


def evaluation_root(args):
    source = args.run_dir.resolve()
    raw = (source / "summary.json").read_bytes()
    original = json.loads(raw)
    split = args.split or original["config"]["eval_split"]
    overrides = {name: getattr(args, name) for name in (
        "model", "data_root", "eval_loops", "eval_limit", "max_length", "max_new_tokens",
        "seed", "truncate_long_prompts", "temperature", "max_new_tokens_by_task")
        if getattr(args, name, None) is not None}
    if args.tasks is not None:
        if not set(args.tasks) <= set(original["config"]["tasks"]):
            raise ValueError("Evaluation tasks must belong to the source training sequence")
        overrides["eval_tasks"] = args.tasks
    if "max_new_tokens_by_task" in overrides and args.max_new_tokens is None:
        overrides["max_new_tokens"] = None
    output = args.output_dir.resolve() if args.output_dir is not None else source
    if output == source:
        if split != original["config"]["eval_split"] or overrides:
            raise ValueError("Changing split or evaluation settings requires a separate --output-dir")
        return source
    protocol = {"source_run": str(source), "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
                "split": split, "checkpoint_scope": args.checkpoint_scope}
    if overrides:
        protocol["overrides"] = overrides
    if (output / "summary.json").exists():
        if json.loads((output / "summary.json").read_text()).get("evaluation_protocol") != protocol:
            raise ValueError("Existing output has a different source/split/checkpoint protocol")
        return output
    planned = checkpoint_plan(original, source)
    if args.checkpoint_scope == "final":
        expected = f"{len(original['config']['tasks']):02d}_after_{original['config']['tasks'][-1]}"
        if not planned or planned[-1][0] != expected:
            raise FileNotFoundError(f"Final training checkpoint is missing: {expected}")
        planned = planned[-1:]
    if not planned:
        raise FileNotFoundError(f"No checkpoints under {source}")
    summary = {key: deepcopy(original[key]) for key in ["config", "method", "training", "precision"] if key in original}
    summary["config"].update(output_dir=str(output), eval_split=split, eval_limit=None,
                             primary_metrics=PRIMARY_METRICS, wandb_mode="disabled")
    summary["config"].update(overrides)
    if "eval_loops" in overrides:
        summary["config"].pop("depth_schedule", None)
    for key in ["model", "data_root"]:
        summary["config"][key] = str((REPO_ROOT / summary["config"][key]).resolve())
    summary.update(evaluation_protocol=protocol, evaluations={}, stage_status={})
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    for checkpoint, _ in planned:
        (output / "checkpoints" / checkpoint).symlink_to(source / "checkpoints" / checkpoint,
                                                         target_is_directory=True)
        summary["stage_status"][checkpoint] = {"evaluation": "pending"}
    save_summary(summary, output)
    return output


def run(args):
    root = evaluation_root(args)
    summary = json.loads((root / "summary.json").read_text())
    planned = checkpoint_plan(summary, root)
    if not planned:
        raise FileNotFoundError(f"No saved checkpoints under {root}")
    if args.checkpoint_scope == "final":
        expected = f"{len(summary['config']['tasks']):02d}_after_{summary['config']['tasks'][-1]}"
        if planned[-1][0] != expected:
            raise FileNotFoundError(f"Final training checkpoint is missing: {expected}")
        planned = planned[-1:]
    for checkpoint, stage in planned:
        summary = json.loads((root / "summary.json").read_text())
        grouped, selected = tasks_by_depth(summary, root, checkpoint, stage)
        for depth, tasks in grouped.items():
            run_in_container("vllm", ["worker", "--run-dir", str(root), "--checkpoint", checkpoint,
                             "--loop-depth", str(depth), "--tasks", *tasks,
                             "--batch-size", str(args.batch_size),
                             "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
        summary = json.loads((root / "summary.json").read_text())
        missing = set(selected) - set(summary.get("evaluations", {}).get(checkpoint, {}))
        if missing:
            raise RuntimeError(f"{checkpoint}: missing evaluations: {sorted(missing)}")
        run_in_container("training", ["report", "--run-dir", str(root), "--checkpoint", checkpoint])
