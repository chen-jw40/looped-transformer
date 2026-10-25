#!/usr/bin/env python3
"""Finalize one vLLM checkpoint evaluation and report its matrix to W&B."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation_reporting import log_evaluation_tables
from ouro_pipeline.training import (
    continual_aggregates, continual_metrics, load_env_file, write_summary,
)
from ouro_pipeline.evaluation import checkpoint_plan


def run(args):
    with (args.run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]
    ordered_names = [name for name, _ in checkpoint_plan(summary, args.run_dir)]
    if args.checkpoint not in ordered_names:
        raise ValueError(f"Unknown checkpoint: {args.checkpoint}")
    stage = int(args.checkpoint[:2])
    through_current = ordered_names[:ordered_names.index(args.checkpoint) + 1]
    evaluations = {name: summary["evaluations"][name] for name in through_current
                   if summary.get("evaluations", {}).get(name)}
    if args.checkpoint not in evaluations:
        raise ValueError(f"{args.checkpoint} has no evaluation results")
    matrix, forgetting = continual_metrics(evaluations, config["tasks"], config.get("primary_metrics"))
    aggregate = continual_aggregates(matrix, forgetting, config["tasks"])
    if len(evaluations) == 1 and len(config["tasks"]) > 1:
        forgetting = {}
        aggregate["backward_transfer"] = None
        aggregate["mean_final_forgetting_from_best"] = None
    status = summary.setdefault("stage_status", {}).setdefault(args.checkpoint, {})
    summary["performance_matrix"] = matrix
    summary["forgetting"] = forgetting
    summary["continual_aggregates"] = aggregate
    status["evaluation"] = "complete"
    for stage_report in summary.get("stages", []):
        if stage_report.get("stage") == stage and stage:
            stage_report["evaluation"] = "complete"
    write_summary(summary, args.run_dir)

    if config.get("wandb_mode", "disabled") != "disabled" and not status.get("wandb_reported"):
        load_env_file(Path(config.get("env_file", ".env")))
        import wandb

        run = wandb.init(
            project=config["wandb_project"], entity=config.get("wandb_entity"),
            name=config.get("wandb_run_name") or args.run_dir.name,
            id=summary.get("wandb_run_id"), resume="allow",
            mode=config["wandb_mode"], dir=str(args.run_dir),
        )
        try:
            log_evaluation_tables(run, summary)
            summary["wandb_run_id"] = run.id
            status["wandb_reported"] = True
            write_summary(summary, args.run_dir)
        finally:
            run.finish()
    print(f"Completed vLLM evaluation for {args.checkpoint}", flush=True)
