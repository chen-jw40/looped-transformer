"""Compare coupled single-seed experiments with existing, protocol-matched runs."""
from __future__ import annotations

import csv
from datetime import datetime
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

from scripts.parameters import REPO_ROOT
from scripts.study_statistics import PROTOCOL_KEYS, score_100


def read_complete(path):
    raw = path.read_bytes() if path.exists() else b""
    summary = json.loads(raw) if raw else {}
    config = summary.get("config", {})
    tasks = config.get("tasks", [])
    complete = bool(tasks) and summary.get("training_status") == "complete"
    for index, task in enumerate(tasks, 1):
        checkpoint = f"{index:02d}_after_{task}"
        results = summary.get("evaluations", {}).get(checkpoint, {})
        complete &= summary.get("stage_status", {}).get(checkpoint, {}).get("evaluation") == "complete"
        complete &= set(results) == set(tasks[:index])
        complete &= set(summary.get("performance_matrix", {}).get(checkpoint, {})) == set(tasks[:index])
        complete &= all(r.get("samples", 0) > 0 for r in results.values())
    return summary, bool(complete), hashlib.sha256(raw).hexdigest() if raw else None


def summarize(summary):
    tasks = summary["config"]["tasks"]
    matrix = summary["performance_matrix"]
    final = f"{len(tasks):02d}_after_{tasks[-1]}"
    scores = {t: score_100(matrix[final][t]) for t in tasks}
    if not all(math.isfinite(v) for row in matrix.values() for v in map(score_100, row.values())):
        raise ValueError("Non-finite evaluation score")
    diagonal = [score_100(matrix[f"{i:02d}_after_{t}"][t]) for i, t in enumerate(tasks, 1)]
    stages = summary["stages"]
    return {**scores, "final_macro_100": statistics.mean(scores.values()),
        "immediate_macro_100": statistics.mean(diagonal),
        "bwt_pp": statistics.mean(scores[t] - diagonal[i] for i, t in enumerate(tasks[:-1])),
        "forgetting_from_best_pp": statistics.mean(
            max(score_100(matrix[f"{j:02d}_after_{tasks[j-1]}"][t]) for j in range(i+1, len(tasks)+1)) - scores[t]
            for i, t in enumerate(tasks)),
        "basis_seconds": sum(s.get("basis_cost", {}).get("total_seconds", 0) for s in stages)
            if any("basis_cost" in s for s in stages) else None,
        "calibration_backwards": sum(s.get("coupling_statistics", {}).get("extra_calibration_backward_passes", 0) for s in stages),
        "final_basis_bytes": stages[-1].get("basis_storage_bytes"),
        "peak_basis_allocated_bytes": max((s.get("basis_cost", {}).get("peak_allocated_bytes") or 0 for s in stages), default=0) or None}


def signature(summary):
    config = summary["config"]
    if config["eval_split"] != "test" or config.get("eval_limit") is not None or config["seed"] != 42:
        raise ValueError("Comparison requires seed42 and full test")
    hashes = {(e["task"], e["split"]): e["sha256"] for e in summary["data_sources"]}
    if set(hashes) != {(t, s) for t in config["tasks"] for s in ("train", "eval", "test")}:
        raise ValueError("Incomplete dataset provenance")
    counts = {f"{c}/{t}": r["samples"] for c, rr in summary["evaluations"].items() for t, r in rr.items()}
    return ({k: config.get(k) for k in PROTOCOL_KEYS}, hashes, counts)


def validate_smoke(run_dir):
    summary, complete, digest = read_complete(run_dir / "summary.json")
    if not complete or len(summary["stages"]) != 2:
        raise RuntimeError("Smoke training and all-stage evaluation must complete")
    for stage in summary["stages"]:
        report = stage["coupling_statistics"]
        assert report["samples"] == report["extra_calibration_backward_passes"] == 4
        assert sum(x["coupled_energy"] for x in report["probe_energy_by_module"].values()) > 0
        assert stage["basis_storage_bytes"] > 0
    diagnostic = summary["training"][1]["projection_diagnostics"]
    assert diagnostic["protected_linear_weights"] > 0
    assert diagnostic["max_WU_history_scaled_error"] < 1e-4
    payload = {"status": "passed", "summary": str(run_dir / "summary.json"),
        "summary_sha256": digest, "two_stage_training_and_evaluation": "complete",
        "rank": summary["config"]["basis_max_rank"], "projection": diagnostic}
    (run_dir.parent / "smoke_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run(root):
    root = Path(root).resolve()
    if root.is_relative_to((REPO_ROOT / "result").resolve()):
        raise ValueError("New comparisons must not write to result/")
    manifest = json.loads((root / "submission.json").read_text())
    with (root / ".comparison.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, statuses, sources, summaries = [], [], [], {}
        reference = None
        for entry in manifest["baselines"] + manifest["jobs"]:
            path = Path(entry["run_dir"]) / "summary.json"
            summary, complete, digest = read_complete(path)
            status = {"variant": entry["variant"], "job_id": entry.get("job_id"),
                "training_status": summary.get("training_status", "not_started"),
                "evaluation_complete": complete, "summary": str(path)}
            statuses.append(status)
            if not complete:
                continue
            current = signature(summary)
            if reference is None:
                reference = current
            if current != reference:
                raise ValueError(f"Training/evaluation protocol, hashes or sample counts differ: {path}")
            config = summary["config"]
            if entry in manifest["jobs"]:
                if config.get("basis_statistic") != "depth_coupled" or config["basis_max_rank"] != entry["rank"]:
                    raise ValueError(f"Wrong coupled experiment: {path}")
                paired = summaries[f"last{entry['rank']}"]["config"]
                for key in ("stat_samples", "stat_vectors_per_call", "stat_vectors_per_layer", "basis_max_rank",
                            "basis_residual_tolerance", "projection_scope"):
                    if config[key] != paired[key]:
                        raise ValueError(f"Unmatched {key}: {path}")
            row = {"variant": entry["variant"], "seed": config["seed"], "rank_cap": config.get("basis_max_rank"),
                   **summarize(summary)}
            rows.append(row)
            summaries[entry["variant"]] = summary
            sources.append({"variant": entry["variant"], "summary": str(path), "sha256": digest,
                "config": config, "data_sources": summary["data_sources"], "evaluation_samples": current[2],
                "status": "complete", "actual_rank_by_stage": [r.get("projection_diagnostics", {}).get("history_union_ranks")
                    for r in summary.get("training", [])]})
        by_name = {r["variant"]: r for r in rows}
        deltas = []
        for job in manifest["jobs"]:
            name, baseline = job["variant"], f"last{job['rank']}"
            if name not in by_name or baseline not in by_name:
                continue
            a, b = by_name[name], by_name[baseline]
            tasks = summaries[name]["config"]["tasks"]
            delta = {"variant": name, "baseline": baseline,
                "final_macro_delta_pp": a["final_macro_100"] - b["final_macro_100"],
                "bwt_delta_pp": a["bwt_pp"] - b["bwt_pp"],
                "immediate_macro_delta_pp": a["immediate_macro_100"] - b["immediate_macro_100"],
                "task_deltas_pp": {t: a[t] - b[t] for t in tasks},
                "tasks_improved": sum(a[t] > b[t] for t in tasks)}
            delta["macro_gain_at_least_2pp"] = delta["final_macro_delta_pp"] >= 2
            deltas.append(delta)
        complete = all(j["variant"] in by_name for j in manifest["jobs"])
        payload = {"status": "complete" if complete else "pending",
            "updated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "score_scale": "0-100; accuracy/ROUGE x100; similarity/SARI unchanged",
            "n_per_variant": 1, "sample_sd_ddof1": None,
            "interpretation": "Single seed42 descriptive comparison, no statistical significance claim. 2pp macro gain is a prespecified descriptive threshold.",
            "precision_difference": "Coupled runs disable TF32 for FP32 projection to enforce WQ accurately; historical activation runs used their original container precision. This pilot does not isolate coupling from that numerical correction.",
            "runs": statuses, "scores": rows, "matched_deltas": deltas, "sources": sources}
        temp = root / ".comparison.json.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temp.replace(root / "comparison.json")
        if rows:
            with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        table = "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in rows[0]) + "</tr>" if rows else "<table>"
        for row in rows:
            table += "<tr>" + "".join(f"<td>{html.escape(f'{v:.3f}' if isinstance(v, float) else str(v))}</td>" for v in row.values()) + "</tr>"
        table += "</table>"
        page = '<!doctype html><meta charset="utf-8"><title>Depth coupling comparison</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;white-space:nowrap}</style>'
        page += f'<h1>深度耦合保护：rank512 / rank128</h1><p>状态：{payload["status"]}。seed42，每组一次；统一0–100尺度。单次结果不能证明统计显著提升。</p>'
        page += '<p>每个新实验与相同rank的last基线匹配采样量；任务后新增校准反向。其他方法为上下文对照，保护预算可能不同。</p>' + table
        page += '<p>新方法的FP32投影禁用TF32以准确保持WQ；旧实验保持其原始精度设置。因此此次试验同时包含这一数值精度修正，不能单独归因于深度交互。</p>'
        for delta in deltas:
            page += f'<p>{delta["variant"]} 对 {delta["baseline"]}：综合分 {delta["final_macro_delta_pp"]:+.3f}，BWT {delta["bwt_delta_pp"]:+.3f}，{delta["tasks_improved"]}/8 个任务提升。</p>'
        (root / "comparison.html").write_text(page)
        print(f"Depth coupling comparison: {payload['status']}; {root / 'comparison.json'}")
        return payload


if __name__ == "__main__":
    run(Path(sys.argv[1]))
