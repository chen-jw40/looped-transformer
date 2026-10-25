#!/usr/bin/env python3
"""Aggregate five completed final-checkpoint test runs on a 0-100 scale."""
from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
import statistics

from scripts.parameters import FINAL_CHECKPOINT, TEST_SOURCES


def run(args):
    root = args.output_root
    with (root / ".aggregate.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = []
        for seed in sorted(TEST_SOURCES):
            path = root / f"seed{seed}" / "summary.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            status = summary.get("stage_status", {}).get(FINAL_CHECKPOINT, {})
            if status.get("evaluation") != "complete" or "test_scores" not in summary:
                if args.if_ready:
                    print(f"Waiting for seed{seed}; five-seed statistics are not ready.")
                    return
                raise RuntimeError(f"Incomplete test run: seed{seed}")
            if summary["config"]["seed"] != seed or summary["config"]["eval_split"] != "test":
                raise ValueError(f"Unexpected seed or split: {path}")
            runs.append(summary)
        # Do not average seeds evaluated with different PBS generation/data settings.
        protocols = []
        for summary in runs:
            protocol = dict(summary["test_protocol"])
            for key in ("source_run", "source_summary_sha256"):
                protocol.pop(key, None)
            protocol["datasets"] = {task: {key: info[key] for key in ("sha256", "samples")}
                                    for task, info in protocol["datasets"].items()}
            protocols.append(protocol)
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise ValueError("Five-seed evaluation protocols differ; use the same PBS settings for every seed")
        tasks = runs[0]["config"]["tasks"]
        rows = []
        for task in [*tasks, "macro_average"]:
            if task == "macro_average":
                values = [s["test_scores"]["macro_score_100"] for s in runs]
                metric, samples = "equal_weight_macro_score", ""
            else:
                results = [s["test_scores"]["tasks"][task] for s in runs]
                if len({(r["metric"], r["samples"]) for r in results}) != 1:
                    raise ValueError(f"Metric or sample count mismatch for {task}")
                values = [r["score_100"] for r in results]
                metric, samples = results[0]["metric"], results[0]["samples"]
            rows.append({"task": task, "metric": metric, "samples_per_seed": samples,
                         **{f"seed{seed}": value for seed, value in zip(sorted(TEST_SOURCES), values)},
                         "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                         "min": min(values), "max": max(values), "n": 5})
        with (root / "test_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "statistics.json").write_text(json.dumps({"split": "test", "seeds": list(sorted(TEST_SOURCES)),
            "checkpoint": FINAL_CHECKPOINT, "score_scale": "0-100", "std_ddof": 1,
            "20Minuten_primary_metric": "sari", "scores": rows}, ensure_ascii=False, indent=2) + "\n")
        lines = ["# 直接全参数 finetune：5 seeds 完整 test 结果", "",
                 "仅评估现有最终 checkpoint；未重新训练，也未追溯执行 early stopping。",
                 "所有任务使用完整 test.json。分数统一为 0–100；标准差为跨种子的样本标准差（ddof=1）。",
                 "20Minuten 使用 SARI，MeetingBank 使用 ROUGE-L，Py150 使用 similarity，其他任务使用 accuracy。",
                 "综合得分为八任务等权平均。仅测试最终 checkpoint，本次没有重新计算全阶段 BWT/遗忘。", "",
                 "| 任务 | 主指标 | seed42 | seed43 | seed44 | seed45 | seed46 | 均值 ± 标准差 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            values = " | ".join(f"{row[f'seed{seed}']:.2f}" for seed in sorted(TEST_SOURCES))
            lines.append(f"| {row['task']} | {row['metric']} | {values} | {row['mean']:.2f} ± {row['sample_std']:.2f} |")
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        macro = rows[-1]
        print(f"Five-seed test macro: {macro['mean']:.4f} ± {macro['sample_std']:.4f}")
