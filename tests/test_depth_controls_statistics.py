import copy
import json

import pytest

from scripts.depth_controls_statistics import run
from tests.test_coupled_statistics import example
from ouro_pipeline.subspace import ACTIVATION_SAMPLING_PROTOCOL


def _study(root):
    last = example()
    last["config"].update(loops=4, eval_loops=4, basis_depth_mode="last")
    trajectory = copy.deepcopy(last)
    trajectory["config"].update(basis_strategy="last_trajectory", basis_depth_mode="all", trajectory_floor="tolerance")
    trajectory["projection_protocol"] = {"extra_calibration_backward_passes": 0}
    trajectory["stages"] = [{"basis_updates": [{"floor_satisfied": True, "new_rank": 2,
        "old_rank": 0, "reference_rank": 2, "last_residual_fraction": .049, "last_residual_limit": .05}]},
        {"basis_updates": [{"floor_satisfied": True, "new_rank": 3, "old_rank": 2,
         "reference_rank": 3, "last_residual_fraction": .05, "last_residual_limit": .05}]}]
    trajectory["performance_matrix"]["02_after_Py150"]["FOMC"]["value"] = .7
    for name, summary in (("last", last), ("trajectory", trajectory)):
        path = root / name
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(summary))
    (root / "submission.json").write_text(json.dumps({
        "baselines": [{"variant": "last", "rank": 128, "run_dir": str(root / "last")}],
        "jobs": [{"variant": "trajectory", "rank": 128, "run_dir": str(root / "trajectory")}],
        "data_manifest": [{"task": t, "split": "test", "samples": 20} for t in last["config"]["tasks"]],
    }))
    return trajectory


def test_single_seed_comparison_scales_and_preserves_provenance(tmp_path):
    _study(tmp_path)
    result = run(tmp_path)
    assert result["status"] == "complete"
    assert result["matched_deltas"][0]["final_macro_delta_pp"] == 5
    assert result["matched_deltas"][0]["bwt_delta_pp"] == 10
    assert result["sample_sd_ddof1"] is None
    assert result["sources"][1]["summary_sha256"]
    assert result["sources"][1]["evaluation_samples"]["02_after_Py150/Py150"] == 20


@pytest.mark.parametrize("mutate", [
    lambda s: s["stages"][0]["basis_updates"][0].update(last_residual_fraction=.06),
    lambda s: s["stages"][0]["basis_updates"][0].update(new_rank=4),
    lambda s: s["config"].update(stat_vectors_per_call=16),
    lambda s: s["evaluations"]["02_after_Py150"]["Py150"].update(samples=19),
    lambda s: s["data_sources"][0].update(sha256="changed"),
])
def test_invalid_rank_floor_sampling_or_data_cannot_claim_matched_results(tmp_path, mutate):
    summary = _study(tmp_path)
    mutate(summary)
    (tmp_path / "trajectory/summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        run(tmp_path)


def test_partial_evaluation_is_pending(tmp_path):
    summary = _study(tmp_path)
    summary["stage_status"]["02_after_Py150"]["evaluation"] = "pending"
    (tmp_path / "trajectory/summary.json").write_text(json.dumps(summary))
    result = run(tmp_path)
    assert result["status"] == "pending"
    assert not result["matched_deltas"]


@pytest.mark.parametrize("gain_scope", ["source_depth_only", "all_depths"])
def test_local_gain_ablation_compares_with_joint_baseline(tmp_path, gain_scope):
    joint = example()
    joint["config"].update(loops=4, eval_loops=4, basis_depth_mode="all",
                           basis_strategy="joint_greedy", history_rank=128)
    local = copy.deepcopy(joint)
    local["config"]["basis_strategy"] = "joint_local"
    local["projection_protocol"] = {"selection_gain_scope": gain_scope}
    local["performance_matrix"]["02_after_Py150"]["FOMC"]["value"] = .7
    for name, summary in [("joint", joint), ("joint_local", local)]:
        path = tmp_path / name
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "submission.json").write_text(json.dumps({
        "baselines": [{"variant": "joint", "rank": 128, "run_dir": str(tmp_path / "joint")}],
        "jobs": [{"variant": "joint_local", "rank": 128, "run_dir": str(tmp_path / "joint_local")}],
        "data_manifest": [{"task": t, "split": "test", "samples": 20} for t in joint["config"]["tasks"]],
    }))
    if gain_scope != "source_depth_only":
        with pytest.raises(ValueError, match="source-depth-only"):
            run(tmp_path)
        return
    result = run(tmp_path)
    assert result["status"] == "complete"
    delta, = result["matched_deltas"]
    assert delta["experiment"] == "joint_local128"
    assert delta["reference"] == "joint128"
    assert delta["final_macro_delta_pp"] == 5
    assert result["sources"][1]["summary_sha256"]


def test_full_baseline_is_compared_without_a_fictitious_rank(tmp_path):
    _study(tmp_path)
    full = json.loads((tmp_path / "last/summary.json").read_text())
    full["config"]["method"] = "full"
    (tmp_path / "full").mkdir()
    (tmp_path / "full/summary.json").write_text(json.dumps(full))
    manifest_path = tmp_path / "submission.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"].append({"variant": "full", "rank": None, "run_dir": str(tmp_path / "full")})
    manifest_path.write_text(json.dumps(manifest))
    result = run(tmp_path)
    assert result["status"] == "complete"
    assert result["scores"][-1]["experiment"] == "full"
    assert len(result["full_baseline_deltas"]) == 2
    assert result["full_baseline_deltas"][0]["final_macro_delta_pp"] == 0


@pytest.mark.parametrize("rank", [64, 128])
@pytest.mark.parametrize("mismatch", [None, "budget", "policy"])
def test_fixed_sampling_study_requires_matching_budget_and_policy(tmp_path, rank, mismatch):
    _study(tmp_path)
    sampling = {"stat_samples": 64, "stat_vectors_per_call": 16, "stat_vectors_per_layer": 1024}
    manifest_path = tmp_path / "submission.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(sampling_config=sampling, activation_sampling=ACTIVATION_SAMPLING_PROTOCOL)
    for entry in manifest["baselines"] + manifest["jobs"]:
        entry["rank"] = rank
        path = tmp_path / ("last" if entry["variant"] == "last" else "trajectory") / "summary.json"
        summary = json.loads(path.read_text())
        summary["config"].update(sampling, basis_max_rank=rank)
        summary.setdefault("projection_protocol", {})["activation_sampling"] = ACTIVATION_SAMPLING_PROTOCOL
        if entry["variant"] == "trajectory":
            if mismatch == "budget":
                summary["config"]["stat_vectors_per_call"] = 2
            elif mismatch == "policy":
                summary["projection_protocol"].pop("activation_sampling")
        path.write_text(json.dumps(summary))
    manifest_path.write_text(json.dumps(manifest))
    if mismatch:
        with pytest.raises(ValueError, match="sampling"):
            run(tmp_path)
    else:
        result = run(tmp_path)
        assert result["status"] == "complete"
        assert result["activation_sampling"] == ACTIVATION_SAMPLING_PROTOCOL
        assert result["sampling_config"] == sampling
