import json

import pytest

from scripts.balanced_statistics import run


def prepare(root, complete):
    jobs = []
    tasks = ["FOMC", "Py150"]
    for variant, strategy, rank in [("b128", "joint_balanced", 128),
                                    ("b512", "joint_balanced", 512)]:
        folder = root / variant
        folder.mkdir()
        jobs.append({"variant": variant, "strategy": strategy, "rank": rank, "run_dir": str(folder)})
        if not complete:
            continue
        matrix = {"01_after_FOMC": {"FOMC": {"metric": "accuracy", "value": .8}},
                  "02_after_Py150": {"FOMC": {"metric": "accuracy", "value": .7},
                                     "Py150": {"metric": "similarity", "value": 60.}}}
        summary = {"config": {"method": "projected", "seed": 42, "tasks": tasks,
                               "basis_strategy": strategy, "basis_max_rank": rank,
                               "history_rank": rank, "eval_split": "test", "eval_limit": None},
                   "training_status": "complete", "performance_matrix": matrix,
                   "stage_status": {c: {"evaluation": "complete"} for c in matrix},
                   "evaluations": {c: {t: {"samples": 10} for t in row} for c, row in matrix.items()},
                   "data_sources": [{"task": t, "split": s, "sha256": "fixture"} for t in tasks for s in ["train", "eval", "test"]],
                   "stages": [{"basis_storage_bytes": 1024}, {"basis_storage_bytes": 2048}]}
        (folder / "summary.json").write_text(json.dumps(summary))
    (root / "submission.json").write_text(json.dumps({"jobs": jobs}))


def test_single_seed_report_scales_metrics_and_preserves_missing_std(tmp_path):
    prepare(tmp_path, True)
    report = run(tmp_path)
    assert report["status"] == "complete" and report["sample_std_ddof1"] is None
    for row in report["scores"]:
        assert row["final_macro_score_100"] == pytest.approx(65)
        assert row["accuracy_tasks_macro_100"] == pytest.approx(70)
        assert row["backward_transfer_pp"] == pytest.approx(-10)
        assert row["forgetting_from_best_pp"] == pytest.approx(5)
    assert len(report["sources"]) == 2
    assert (tmp_path / "comparison.csv").is_file()


def test_report_does_not_turn_pending_runs_into_results(tmp_path):
    prepare(tmp_path, False)
    report = run(tmp_path)
    assert report["status"] == "pending" and report["scores"] == []
    assert not (tmp_path / "comparison.csv").exists()
    assert all(not r["evaluation_complete"] for r in report["runs"])


def test_report_rejects_mismatched_test_counts(tmp_path):
    prepare(tmp_path, True)
    path = tmp_path / "b512/summary.json"
    summary = json.loads(path.read_text())
    summary["evaluations"]["02_after_Py150"]["Py150"]["samples"] = 9
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sample counts differ"):
        run(tmp_path)
