import copy
import json
from pathlib import Path

import pytest

from scripts import matched_baselines
from scripts.parameters import REPO_ROOT


@pytest.mark.parametrize("wrong_budget", [False, True])
def test_comparison_keeps_full_single_seed_and_matches_only_seed42(tmp_path, wrong_budget):
    source = REPO_ROOT / "runs/20261017_fixed_epochs_mb8_depth_study/full/seed42/summary.json"
    if not source.exists():
        pytest.skip("Completed fixed-epoch Full fixture unavailable")
    summary, complete, digest = matched_baselines.read_complete(source)
    assert complete
    data = REPO_ROOT / "runs/20261024_joint_local128_5seeds/data_manifest.json"
    (tmp_path / "data_manifest.json").write_bytes(data.read_bytes())
    reference_config = tmp_path / "full_config.json"
    reference_config.write_text(json.dumps(summary["config"]))
    manifest = dict(method="lora", score_scale="0-100", baselines=[dict(method="full", seed=42,
        run_dir=str(source.parent), summary_sha256=digest, resolved_config=str(reference_config))], jobs=[])
    for index, seed in enumerate(range(42, 47)):
        output = tmp_path / f"fixture_seed{seed}"
        output.mkdir()
        local = copy.deepcopy(summary)
        local["config"].update(method="lora", seed=seed, output_dir=str(output))
        local["performance_matrix"]["08_after_20Minuten"]["FOMC"]["value"] += .008 * (index + 1)
        if wrong_budget and seed == 45:
            local["training"][0]["epochs_requested"] = 1
        config = tmp_path / f"config_seed{seed}.json"
        config.write_text(json.dumps(local["config"]))
        (output / "summary.json").write_text(json.dumps(local))
        manifest["jobs"].append(dict(method="lora", seed=seed, run_dir=str(output), resolved_config=str(config)))
    (tmp_path / "submission.json").write_text(json.dumps(manifest))
    if wrong_budget:
        with pytest.raises(ValueError, match="training budget"):
            matched_baselines.compare(tmp_path)
    else:
        result = matched_baselines.compare(tmp_path)
        assert result["n"] == 5 and result["status"] == "complete"
        assert result["references"]["full"]["n"] == 1
        assert result["references"]["full"]["statistics"]["final_macro_100"]["sample_sd"] is None
        assert result["paired_delta_statistics"]["full"]["n"] == 1
        assert result["paired_deltas"]["full"][0]["seed"] == 42
        assert result["paired_deltas"]["full"][0]["final_macro_100"] == pytest.approx(.1)
        assert result["statistics"]["final_macro_100"]["sample_sd"] == pytest.approx(.158113883)
