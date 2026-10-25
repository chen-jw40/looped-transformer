import json
from pathlib import Path

import pytest

from scripts.parameters import parse_args, training_config
from scripts.study_statistics import run


def make_study(root):
    tasks = ["FOMC", "Py150"]
    for method in ("full", "osft"):
        for seed in (42, 43):
            path = root / method / f"seed{seed}"
            path.mkdir(parents=True)
            config = training_config(parse_args(["train", "--method", method, "--output-dir", str(path),
                                                 "--tasks", *tasks, "--seed", str(seed), "--checkpoint-scope", "all"]))
            first, final = "01_after_FOMC", "02_after_Py150"
            accuracy = 0.6 + 0.1 * (seed - 42)
            summary = {"config": config, "training_status": "complete",
                       "data_sources": [{"task": t, "split": split, "sha256": t + split, "path": t + split}
                                        for t in tasks for split in ("train", "eval", "test")],
                       "stage_status": {name: {"evaluation": "complete"} for name in (first, final)},
                       "evaluations": {first: {"FOMC": {"samples": 100}},
                                       final: {"FOMC": {"samples": 100}, "Py150": {"samples": 80}}},
                       "performance_matrix": {first: {"FOMC": {"metric": "accuracy", "value": 0.8}},
                                              final: {"FOMC": {"metric": "accuracy", "value": accuracy},
                                                      "Py150": {"metric": "similarity", "value": 80}}}}
            (path / "summary.json").write_text(json.dumps(summary))


def test_mixed_scales_bwt_sample_std_and_source_provenance(tmp_path):
    make_study(tmp_path)
    run(parse_args(["summarize-study", "--output-root", str(tmp_path), "--seeds", "42", "43"]))
    statistics = json.loads((tmp_path / "statistics.json").read_text())
    rows = {row["metric"]: row for row in statistics["scores"] if row["method"] == "osft"}
    assert rows["final_macro_average"]["mean"] == pytest.approx(72.5)
    assert rows["backward_transfer"]["mean"] == pytest.approx(-15)
    assert rows["final_macro_average"]["sample_std"] == pytest.approx(5 / 2**0.5)
    assert statistics["std_ddof"] == 1
    assert len(statistics["sources"]) == 4
    assert all(len(source["summary_sha256"]) == 64 for source in statistics["sources"])
    assert (tmp_path / "scores.csv").exists()


@pytest.mark.parametrize("change", ["protocol", "hash", "samples", "incomplete"])
def test_mismatched_or_incomplete_runs_are_never_averaged(tmp_path, change):
    make_study(tmp_path)
    path = tmp_path / "osft/seed43/summary.json"
    summary = json.loads(path.read_text())
    if change == "protocol":
        summary["config"]["early_stopping"] = not summary["config"]["early_stopping"]
    elif change == "hash":
        summary["data_sources"][0]["sha256"] = "different"
    elif change == "samples":
        summary["evaluations"]["01_after_FOMC"]["FOMC"]["samples"] = 1
    else:
        summary["stage_status"]["01_after_FOMC"]["evaluation"] = "pending"
    path.write_text(json.dumps(summary))
    with pytest.raises((ValueError, RuntimeError)):
        run(parse_args(["summarize-study", "--output-root", str(tmp_path), "--seeds", "42", "43"]))
    assert not (tmp_path / "statistics.json").exists()


def test_full_olora_comparison_keeps_algorithm_provenance_and_forgetting(tmp_path):
    make_study(tmp_path)
    (tmp_path / "osft").rename(tmp_path / "olora")
    for seed in (42, 43):
        path = tmp_path / "olora" / f"seed{seed}" / "summary.json"
        summary = json.loads(path.read_text())
        summary["config"]["method"] = "olora"
        summary["olora_protocol"] = {"reference_commit": "pinned", "rank_per_task": 8}
        path.write_text(json.dumps(summary))
    run(parse_args(["summarize-study", "--output-root", str(tmp_path),
                    "--seeds", "42", "43", "--methods", "full", "olora"]))
    payload = json.loads((tmp_path / "statistics.json").read_text())
    assert payload["methods"] == ["full", "olora"]
    row = next(r for r in payload["scores"]
               if r["method"] == "olora" and r["metric"] == "mean_final_forgetting_from_best")
    assert row["mean"] == pytest.approx(7.5)
    assert all(s["olora_protocol"]["reference_commit"] == "pinned"
               for s in payload["sources"] if s["method"] == "olora")
