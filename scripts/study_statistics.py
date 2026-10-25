"""Compare matched continual-learning runs, preserving sources and score scales."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from pathlib import Path
import statistics

from scripts.parameters import REPO_ROOT


PROTOCOL_KEYS = ("model", "data_root", "tasks", "eval_tasks", "loops", "eval_loops",
                 "train_limit", "epochs", "epochs_by_task", "learning_rate", "weight_decay", "micro_batch_size",
                 "gradient_accumulation", "max_length", "truncate_long_prompts", "early_stopping",
                 "early_stopping_steps", "early_stopping_max_evals", "early_stopping_samples",
                 "early_stopping_seed", "early_stopping_patience", "early_stopping_min_delta",
                 "eval_split", "eval_limit", "temperature", "max_new_tokens", "max_new_tokens_by_task",
                 "primary_metrics")


def score_100(result):
    return result["value"] * (1 if result["metric"] in {"similarity", "sari"} else 100)


def run(args):
    root = args.output_root.resolve()
    if root.is_relative_to((REPO_ROOT / "result").resolve()):
        raise ValueError("New study statistics must not write to result/")
    if not root.is_dir():
        raise FileNotFoundError(root)
    with (root / ".study_statistics.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs, sources, protocol, counts, hashes = {}, [], None, None, None
        for method in args.methods:
            runs[method] = []
            ewc_protocol = None
            for seed in args.seeds:
                path = root / method / f"seed{seed}" / "summary.json"
                raw = path.read_bytes() if path.exists() else b""
                summary = json.loads(raw) if raw else {}
                config = summary.get("config", {})
                tasks = config.get("tasks", [])
                complete = bool(tasks) and summary.get("training_status") == "complete"
                for index, task in enumerate(tasks, 1):
                    name = f"{index:02d}_after_{task}"
                    complete &= summary.get("stage_status", {}).get(name, {}).get("evaluation") == "complete"
                    complete &= set(summary.get("evaluations", {}).get(name, {})) == set(tasks[:index])
                if not complete:
                    if args.if_ready:
                        print(f"Study statistics pending: {path}")
                        return
                    raise RuntimeError(f"Incomplete study run: {path}")
                if config["seed"] != seed or config["method"] != method or config["eval_split"] != "test":
                    raise ValueError(f"Wrong method, seed or split: {path}")
                if method == "ewc":
                    current_ewc = {key: config.get(key) for key in ("ewc_lambda", "ewc_fisher_samples")}
                    current_ewc["variant"] = summary.get("ewc_protocol", {}).get("variant")
                    if ewc_protocol is None:
                        ewc_protocol = current_ewc
                    elif current_ewc != ewc_protocol:
                        raise ValueError(f"EWC protocols differ: {path}")
                current_protocol = {key: config.get(key) for key in PROTOCOL_KEYS}
                current_hashes = {(entry["task"], entry["split"]): entry["sha256"]
                                  for entry in summary["data_sources"]}
                expected_hash_keys = {(task, split) for task in tasks for split in ("train", "eval", "test")}
                if set(current_hashes) != expected_hash_keys:
                    raise ValueError(f"Missing data source hashes: {path}")
                current_counts = {(name, task): result["samples"] for name, results in summary["evaluations"].items()
                                  for task, result in results.items()}
                if protocol is None:
                    protocol, hashes, counts = current_protocol, current_hashes, current_counts
                elif (protocol, hashes, counts) != (current_protocol, current_hashes, current_counts):
                    raise ValueError(f"Training/evaluation protocols, dataset hashes or sample counts differ: {path}")
                if config.get("eval_limit") is not None:
                    raise ValueError("Formal study statistics require full test evaluation")
                final = f"{len(tasks):02d}_after_{tasks[-1]}"
                matrix = summary["performance_matrix"]
                values = {task: score_100(matrix[final][task]) for task in tasks}
                diagonal = [score_100(matrix[f"{i:02d}_after_{task}"][task])
                            for i, task in enumerate(tasks, 1)]
                values["final_macro_average"] = statistics.mean(values.values())
                values["immediate_task_macro_average"] = statistics.mean(diagonal)
                if len(tasks) < 2:
                    raise ValueError("BWT requires at least two tasks")
                values["backward_transfer"] = statistics.mean(
                    values[task] - diagonal[i] for i, task in enumerate(tasks[:-1]))
                values["mean_final_forgetting_from_best"] = statistics.mean(
                    max(score_100(matrix[f"{j:02d}_after_{tasks[j-1]}"][task])
                        for j in range(i + 1, len(tasks) + 1)) - values[task]
                    for i, task in enumerate(tasks))
                runs[method].append(values)
                sources.append({"method": method, "seed": seed, "summary": str(path),
                                "summary_sha256": hashlib.sha256(raw).hexdigest(),
                                "training_status": summary["training_status"], "evaluation_status": "complete",
                                "samples": {f"{name}/{task}": count for (name, task), count in current_counts.items()},
                                "data_sources": summary["data_sources"],
                                "config": config, "recovery_history": summary.get("recovery_history", []),
                                "osft_protocol": summary.get("osft_protocol"),
                                "olora_protocol": summary.get("olora_protocol"),
                                "ewc_protocol": summary.get("ewc_protocol")})
        rows = []
        for method, results in runs.items():
            for metric in results[0]:
                values = [result[metric] for result in results]
                rows.append({"method": method, "metric": metric, "n": len(values),
                             "unit": "percentage points" if metric in {"backward_transfer", "mean_final_forgetting_from_best"} else "0-100",
                             **{f"seed{seed}": value for seed, value in zip(args.seeds, values)},
                             "mean": statistics.mean(values), "sample_std": statistics.stdev(values)})
        with (root / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload = {"status": "complete", "split": "test", "seeds": args.seeds, "std_ddof": 1,
                   "methods": args.methods, "protocol": protocol, "scores": rows, "sources": sources}
        (root / "statistics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        lines = ["# Ouro " + " / ".join(args.methods) + "：匹配协议复现", "",
                 "所有阶段完整 test；混合指标统一到 0–100。样本标准差 ddof=1。",
                 "各方法的实际配置与来源保存在 statistics.json。", "",
                 "| 方法 | 指标 | 均值 | 样本标准差 | n | 单位 |",
                 "|---|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['method']} | {row['metric']} | {row['mean']:.4f} | {row['sample_std']:.4f} | {row['n']} | {row['unit']} |")
        lines += ["", "逐 seed 分数见 scores.csv；来源路径、哈希、配置和样本数见 statistics.json。"]
        (root / "summary.md").write_text("\n".join(lines) + "\n")
        print(f"Matched study statistics saved to {root}")
