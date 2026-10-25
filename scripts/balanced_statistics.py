"""Collect the retained single-seed depth-balance experiments after PBS evaluation."""
from __future__ import annotations

import csv
from datetime import datetime
import fcntl
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

from scripts.parameters import REPO_ROOT
from scripts.study_statistics import PROTOCOL_KEYS, score_100


def run(root):
    root = Path(root).resolve()
    if root.is_relative_to((REPO_ROOT / "result").resolve()):
        raise ValueError("New study statistics must not write to result/")
    submission = json.loads((root / "submission.json").read_text())
    with (root / ".balanced_statistics.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, sources, statuses = [], [], []
        common = None
        for job in submission["jobs"]:
            path = Path(job["run_dir"]) / "summary.json"
            raw = path.read_bytes() if path.exists() else b""
            summary = json.loads(raw) if raw else {}
            config = summary.get("config", {})
            tasks = config.get("tasks", [])
            complete = bool(tasks) and summary.get("training_status") == "complete"
            for i, task in enumerate(tasks, 1):
                checkpoint = f"{i:02d}_after_{task}"
                complete &= summary.get("stage_status", {}).get(checkpoint, {}).get("evaluation") == "complete"
                complete &= set(summary.get("evaluations", {}).get(checkpoint, {})) == set(tasks[:i])
            statuses.append({"variant": job["variant"], "job_id": job.get("job_id"),
                             "training_status": summary.get("training_status", "not_started"),
                             "evaluation_complete": bool(complete), "summary": str(path)})
            if not complete:
                continue
            if (config["seed"] != 42 or config["method"] != "projected"
                    or config["basis_strategy"] != job["strategy"]
                    or config["basis_max_rank"] != job["rank"] or config["history_rank"] != job["rank"]
                    or config["eval_split"] != "test" or config.get("eval_limit") is not None):
                raise ValueError(f"Unexpected experiment configuration: {path}")
            hashes = {(e["task"], e["split"]): e["sha256"] for e in summary["data_sources"]}
            if set(hashes) != {(t, s) for t in tasks for s in ["train", "eval", "test"]}:
                raise ValueError(f"Incomplete data provenance: {path}")
            counts = {(checkpoint, task): r["samples"] for checkpoint, results in summary["evaluations"].items()
                      for task, r in results.items()}
            protocol = {k: config.get(k) for k in PROTOCOL_KEYS}
            signature = (protocol, hashes, counts)
            if common is not None and common != signature:
                raise ValueError("Experiment protocols, data hashes or evaluation sample counts differ")
            common = signature
            matrix = summary["performance_matrix"]
            final = f"{len(tasks):02d}_after_{tasks[-1]}"
            scores = {t: score_100(matrix[final][t]) for t in tasks}
            diagonal = [score_100(matrix[f"{i:02d}_after_{t}"][t]) for i, t in enumerate(tasks, 1)]
            accuracy = [v for t, v in scores.items() if matrix[final][t]["metric"] == "accuracy"]
            stages = summary["stages"]
            row = {"variant": job["variant"], "seed": 42, "rank": job["rank"], **scores,
                   "final_macro_score_100": statistics.mean(scores.values()),
                   "accuracy_tasks_macro_100": statistics.mean(accuracy) if accuracy else None,
                   "immediate_macro_score_100": statistics.mean(diagonal),
                   "backward_transfer_pp": statistics.mean(scores[t] - diagonal[i] for i, t in enumerate(tasks[:-1])),
                   "forgetting_from_best_pp": statistics.mean(
                       max(score_100(matrix[f"{j:02d}_after_{tasks[j-1]}"][t]) for j in range(i+1, len(tasks)+1)) - scores[t]
                       for i, t in enumerate(tasks)),
                   "basis_collection_seconds": sum(s.get("basis_cost", {}).get("collection_seconds", 0) for s in stages),
                   "basis_total_seconds": sum(s.get("basis_cost", {}).get("total_seconds", 0) for s in stages),
                   "final_basis_storage_bytes": stages[-1]["basis_storage_bytes"]}
            rows.append(row)
            sources.append({"variant": job["variant"], "summary": str(path),
                            "summary_sha256": hashlib.sha256(raw).hexdigest(), "config": config,
                            "data_sources": summary["data_sources"],
                            "evaluation_samples": {f"{c}/{t}": n for (c, t), n in counts.items()},
                            "completion": "training and all-stage full test complete"})
        complete = bool(submission["jobs"]) and len(rows) == len(submission["jobs"])
        payload = {"status": "complete" if complete else "pending", "updated_at": datetime.now().astimezone().isoformat(),
                   "seed": 42, "n_per_variant": 1, "sample_std_ddof1": None,
                   "score_scale": "0-100; accuracy/ROUGE multiplied by 100, similarity/SARI unchanged",
                   "note": "Single-seed descriptive comparison; no multi-seed SD or significance claim.",
                   "runs": statuses, "scores": rows, "sources": sources}
        temp = root / ".comparison.json.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temp.replace(root / "comparison.json")
        if rows:
            with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        headers = list(rows[0]) if rows else ["variant", "training_status", "evaluation_complete"]
        table_rows = rows if rows else statuses
        table = '<table><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in headers) + '</tr>'
        for row in table_rows:
            table += '<tr>' + ''.join('<td>' + html.escape(f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k])) + '</td>' for k in headers) + '</tr>'
        table += '</table>'
        page = '<!doctype html><meta charset="utf-8"><title>Depth balance comparison</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;white-space:nowrap}</style>'
        page += '<h1>深度覆盖均衡：rank 128 / 512</h1><p>seed 42，每组一次；无多 seed 标准差。混合指标统一到 0–100，综合分不是平均准确率。</p>'
        page += '<p>完成状态：' + payload['status'] + '</p>' + table
        (root / "comparison.html").write_text(page)
        print(f"Balanced study: {len(rows)}/{len(submission['jobs'])} runs complete; {root / 'comparison.json'}")
    return payload


if __name__ == "__main__":
    run(sys.argv[1])
