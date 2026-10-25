"""Report the single-seed, forward-only depth controls without mixing budgets."""
from __future__ import annotations

import csv
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.coupled_statistics import read_complete, signature, summarize
from scripts.parameters import REPO_ROOT


def validate_variant(summary, entry, sampling_config=None, activation_sampling=None):
    config = summary["config"]
    rank, variant = entry["rank"], entry["variant"]
    if variant == "full":
        if config.get("method") != "full" or rank is not None:
            raise ValueError("Full baseline must be unconstrained, without a protection rank")
        return
    if config["basis_max_rank"] != rank or config.get("basis_statistic", "activation") != "activation":
        raise ValueError("Wrong protection rank or statistic")
    sampling_config = sampling_config or {
        "stat_samples": 64, "stat_vectors_per_call": rank // 32, "stat_vectors_per_layer": rank * 2,
    }
    if any(config[key] != expected for key, expected in {
        "loops": 4, "eval_loops": 4, "seed": 42, **sampling_config,
        "basis_residual_tolerance": .05, "projection_scope": "all-linear",
    }.items()):
        raise ValueError("Unmatched depth-control sampling or training protocol")
    recorded_sampling = summary.get("projection_protocol", {}).get("activation_sampling")
    if recorded_sampling != activation_sampling:
        raise ValueError("Activation sampling policy differs from the study manifest")
    strategy = config.get("basis_strategy", "append")
    if variant == "trajectory":
        if strategy != "last_trajectory" or config["basis_depth_mode"] != "all" or config["trajectory_floor"] != "tolerance":
            raise ValueError("Wrong trajectory configuration")
        if summary["projection_protocol"]["extra_calibration_backward_passes"] != 0:
            raise ValueError("Trajectory experiments must be forward-only")
        for stage in summary["stages"]:
            for report in stage["basis_updates"]:
                if (not report["floor_satisfied"] or report["new_rank"] != report["reference_rank"]
                        or report["new_rank"] < report["old_rank"]
                        or report["last_residual_fraction"] > report["last_residual_limit"] + 2e-7):
                    raise ValueError("Trajectory violated last floor, reference rank or append-only history")
    elif variant == "first":
        if strategy != "append" or config["basis_depth_mode"] != "task" or config["route_depths"] != [0] * len(config["tasks"]):
            raise ValueError("Every task must use first-loop history")
        if any(stage["basis_source_depths"] != [0] for stage in summary["stages"]):
            raise ValueError("Wrong first-loop sampling depth")
    elif variant == "last":
        if strategy != "append" or config["basis_depth_mode"] != "last":
            raise ValueError("Wrong last baseline")
    elif variant in {"joint", "joint_local"}:
        expected_strategy = "joint_greedy" if variant == "joint" else "joint_local"
        if strategy != expected_strategy or config["basis_depth_mode"] != "all" or config["history_rank"] != rank:
            raise ValueError("Wrong joint baseline or history rank")
        if variant == "joint_local" and summary.get("projection_protocol", {}).get("selection_gain_scope") != "source_depth_only":
            raise ValueError("Local ablation must record source-depth-only selection gains")
    else:
        raise ValueError(f"Unknown depth-control variant: {variant}")


def run(root):
    root = Path(root).resolve()
    if root.is_relative_to((REPO_ROOT / "result").resolve()):
        raise ValueError("New reports must not write to result/")
    manifest = json.loads((root / "submission.json").read_text())
    with (root / ".comparison.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, statuses, sources, summaries = [], [], [], {}
        common = None
        for entry in manifest["baselines"] + manifest["jobs"]:
            path = Path(entry["run_dir"]) / "summary.json"
            summary, complete, digest = read_complete(path)
            name = "full" if entry["variant"] == "full" else f"{entry['variant']}{entry['rank']}"
            statuses.append({"experiment": name, "job_id": entry.get("job_id"),
                "training_status": summary.get("training_status", "not_started"),
                "all_stage_full_test_complete": complete, "summary": str(path)})
            if not complete:
                continue
            if entry.get("resolved_config"):
                expected_config = json.loads(Path(entry["resolved_config"]).read_text())
                if summary["config"] != expected_config:
                    raise ValueError(f"Run differs from frozen study configuration: {path}")
            if manifest.get("fixed_epoch_budget"):
                config = summary["config"]
                for report in summary["training"]:
                    epochs = manifest["fixed_epoch_budget"][report["task"]]
                    micro_per_epoch = math.ceil(report["samples"] / config["micro_batch_size"])
                    updates_per_epoch = math.ceil(micro_per_epoch / config["gradient_accumulation"])
                    if (report.get("epochs_requested") != epochs
                            or report["micro_steps"] != micro_per_epoch * epochs
                            or report["optimizer_steps"] != updates_per_epoch * epochs
                            or config["early_stopping"] or "early_stopping" in report):
                        raise ValueError(f"Incomplete fixed-epoch training budget: {path}")
            current = signature(summary)
            if common is None:
                common = current
            if current != common:
                raise ValueError(f"Training protocol, dataset hashes or evaluation samples differ: {path}")
            validate_variant(summary, entry, manifest.get("sampling_config"), manifest.get("activation_sampling"))
            counts = {e["task"]: e["samples"] for e in manifest["data_manifest"] if e["split"] == "test"}
            for results in summary["evaluations"].values():
                if any(r["samples"] != counts[t] for t, r in results.items()):
                    raise ValueError(f"Evaluation is not complete test: {path}")
            summaries[name] = summary
            rows.append({"experiment": name, "variant": entry["variant"], "rank": entry["rank"],
                "seed": 42, **summarize(summary)})
            sources.append({"experiment": name, "summary": str(path), "summary_sha256": digest,
                "status": "complete", "split": "test", "config": summary["config"],
                "data_sources": summary["data_sources"], "evaluation_samples": current[2],
                "actual_rank_by_stage": [r.get("projection_diagnostics", {}).get("history_union_ranks", {})
                                         for r in summary["training"]]})
        lookup = {r["experiment"]: r for r in rows}
        deltas = []
        for name, row in lookup.items():
            reference = (f"joint{row['rank']}" if row["variant"] == "joint_local" else
                         "full" if row["variant"] == "last" else f"last{row['rank']}")
            if name == reference or reference not in lookup:
                continue
            baseline = lookup[reference]
            deltas.append({"experiment": name, "reference": reference,
                "final_macro_delta_pp": row["final_macro_100"] - baseline["final_macro_100"],
                "immediate_macro_delta_pp": row["immediate_macro_100"] - baseline["immediate_macro_100"],
                "bwt_delta_pp": row["bwt_pp"] - baseline["bwt_pp"],
                "task_deltas_pp": {t: row[t] - baseline[t] for t in summaries[name]["config"]["tasks"]}})
        full_deltas = [{"experiment": name, "reference": "full",
            "final_macro_delta_pp": row["final_macro_100"] - lookup["full"]["final_macro_100"],
            "immediate_macro_delta_pp": row["immediate_macro_100"] - lookup["full"]["immediate_macro_100"],
            "bwt_delta_pp": row["bwt_pp"] - lookup["full"]["bwt_pp"]}
            for name, row in lookup.items() if name != "full"] if "full" in lookup else []
        payload = {"status": "complete" if all(s["all_stage_full_test_complete"] for s in statuses) else "pending",
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "score_scale": "0-100; accuracy/ROUGE x100; Py150 similarity/SARI unchanged",
            "n_per_variant": 1, "sample_sd_ddof1": None,
            "interpretation": "One seed42 per variant; descriptive results only. Requested rank is a cap; consult actual_rank_by_stage for the protected rank actually used.",
            "sampling_budget": "first/last: one depth; trajectory/joint/joint_local: same N per depth at four depths, four times transient activation records",
            "sampling_config": manifest.get("sampling_config"),
            "activation_sampling": manifest.get("activation_sampling"),
            "runs": statuses, "scores": rows, "matched_deltas": deltas,
            "full_baseline_deltas": full_deltas, "sources": sources}
        temporary = root / ".comparison.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(root / "comparison.json")
        if rows:
            temporary_csv = root / ".scores.csv.tmp"
            with temporary_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary_csv.replace(root / "scores.csv")
        print(f"Depth controls: {sum(s['all_stage_full_test_complete'] for s in statuses)}/{len(statuses)} full-test runs complete", flush=True)
        return payload
