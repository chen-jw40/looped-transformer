import copy
import json

import pytest

from scripts.coupled_statistics import run, summarize


def example():
    tasks = ["FOMC", "Py150"]
    matrix = {"01_after_FOMC": {"FOMC": {"metric": "accuracy", "value": .8}},
        "02_after_Py150": {"FOMC": {"metric": "accuracy", "value": .6},
                            "Py150": {"metric": "similarity", "value": 40}}}
    return {"config": {"tasks": tasks, "eval_tasks": tasks, "seed": 42, "method": "projected",
                "eval_split": "test", "eval_limit": None, "basis_max_rank": 128,
                "stat_samples": 64, "stat_vectors_per_call": 4, "stat_vectors_per_layer": 256,
                "basis_residual_tolerance": .05, "projection_scope": "all-linear"},
        "training_status": "complete", "performance_matrix": matrix,
        "stages": [{}, {}], "training": [],
        "data_sources": [{"task": t, "split": s, "sha256": t + s} for t in tasks for s in ("train", "eval", "test")],
        "stage_status": {c: {"evaluation": "complete"} for c in matrix},
        "evaluations": {c: {t: {"samples": 20} for t in row} for c, row in matrix.items()}}


def setup_study(root, mismatch=None):
    baseline, candidate = example(), example()
    candidate["config"]["basis_statistic"] = "depth_coupled"
    candidate["performance_matrix"]["02_after_Py150"]["FOMC"]["value"] = .7
    if mismatch:
        mismatch(candidate)
    for name, summary in (("last128", baseline), ("coupled128", candidate)):
        path = root / name
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(summary))
    manifest = {"baselines": [{"variant": "last128", "run_dir": str(root / "last128")}],
        "jobs": [{"variant": "coupled128", "run_dir": str(root / "coupled128"), "rank": 128}]}
    (root / "submission.json").write_text(json.dumps(manifest))


def test_scales_deltas_and_source_provenance(tmp_path):
    setup_study(tmp_path)
    payload = run(tmp_path)
    assert payload["status"] == "complete"
    assert payload["scores"][0]["final_macro_100"] == 50
    assert payload["matched_deltas"][0]["final_macro_delta_pp"] == 5
    assert payload["matched_deltas"][0]["bwt_delta_pp"] == 10
    assert payload["sample_sd_ddof1"] is None
    assert payload["sources"][0]["sha256"]


@pytest.mark.parametrize("mismatch", [
    lambda s: s["config"].update(stat_vectors_per_call=8),
    lambda s: s["evaluations"]["02_after_Py150"]["FOMC"].update(samples=19),
    lambda s: s["data_sources"][0].update(sha256="changed"),
    lambda s: s["config"].update(eval_limit=20),
])
def test_unmatched_protocol_cannot_claim_comparison(tmp_path, mismatch):
    setup_study(tmp_path, mismatch)
    with pytest.raises(ValueError):
        run(tmp_path)
    assert not (tmp_path / "comparison.json").exists()


def test_incomplete_evaluation_remains_pending(tmp_path):
    setup_study(tmp_path, lambda s: s["stage_status"]["02_after_Py150"].update(evaluation="pending"))
    payload = run(tmp_path)
    assert payload["status"] == "pending" and not payload["matched_deltas"]
