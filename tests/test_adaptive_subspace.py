import os
import json
import subprocess
from pathlib import Path

import pytest
import torch

from ouro_pipeline.adaptive_subspace import AdaptiveBasisStore, allocate_joint, update_history
from ouro_pipeline.projected_full import HistoricalFullWeightProjection
from scripts.parameters import parse_args, training_config
from tests.test_projected_full import TinyOuro


def covariance(h):
    return (h["basis"] * h["eigenvalues"]) @ h["basis"].T


def test_history_matches_task_equal_normalized_second_moments():
    torch.manual_seed(31)
    xs = [torch.randn(8, 5), 100 * torch.randn(13, 5), torch.randn(7, 5)]
    h = None
    for count, x in enumerate(xs, 1):
        h = update_history(h, x, 5, "exact")
        expected = sum(y.T @ y / y.square().sum() for y in xs[:count]) / count
        torch.testing.assert_close(covariance(h), expected, atol=1e-6, rtol=1e-5)
    assert h["task_count"] == 3 and h["full_trace"] == 1


def test_compression_does_not_renormalize_retained_energy():
    h = update_history(None, torch.eye(4), 1, "exact")
    assert h["full_trace"] == 1
    assert float(h["eigenvalues"].sum()) == pytest.approx(.25)
    h2 = update_history(h, torch.eye(4), 1, "exact")
    assert float(h2["eigenvalues"].sum()) == pytest.approx(.25)


@pytest.mark.parametrize("selector", ["joint_greedy", "joint_local", "joint_pca", "joint_balanced"])
def test_duplicates_only_cost_one_direction_and_projection_is_orthogonal(selector):
    h = update_history(None, torch.tensor([[1., 0., 0.]]), 3, "exact")
    q, report = allocate_joint([h] * 4, 3, selector, method="exact")
    assert q.shape == (3, 1)
    assert report["coverage"] == pytest.approx([1.] * 4)
    torch.testing.assert_close(q.T @ q, torch.eye(1))


def test_joint_greedy_scores_against_all_depths_not_just_source():
    eye = torch.eye(3)
    def history(u, energy):
        return {"basis": u, "eigenvalues": torch.tensor(energy), "full_trace": 1.}
    # e0 has .5 in three depths (total 1.5); e2 has 1 in one depth.
    histories = [history(eye[:, :2], [.5, .5])] * 3 + [history(eye[:, 2:], [1.])]
    q, report = allocate_joint(histories, 1)
    assert report["joint_coverage"] == pytest.approx(.375)
    assert float(q[2].abs().max()) == 0


def test_pca_attains_spectral_optimum_and_bounds_greedy_coverage():
    torch.manual_seed(29)
    histories = [update_history(None, torch.randn(20, 8), 6, "exact") for _ in range(4)]
    c = sum(covariance(h) for h in histories) / 4
    optimum = torch.linalg.eigvalsh(c)[-3:].sum()
    q, pca = allocate_joint(histories, 3, "joint_pca", method="exact")
    g, greedy = allocate_joint(histories, 3, "joint_greedy")
    assert pca["joint_coverage"] == pytest.approx(float(optimum), abs=1e-6)
    assert greedy["joint_coverage"] <= pca["joint_coverage"] + 1e-6
    for basis in (q, g):
        torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("selector", ["joint_greedy", "joint_local", "joint_pca"])
def test_history_can_replace_saturated_basis_and_round_trip(tmp_path, selector):
    store = AdaptiveBasisStore(2, selector, "exact")
    key = "layers.0.self_attn.q_proj@ut0"
    store.update({key: torch.tensor([[1., 0., 0.]])}, 1)
    for _ in range(2):
        store.update({key: torch.tensor([[0., 1., 0.]])}, 1)
    q = store.bases[key.split("@ut")[0]]
    torch.testing.assert_close(q @ q.T, torch.diag(torch.tensor([0., 1., 0.])))
    path = tmp_path / "state.pt"
    store.save(path)
    restored = AdaptiveBasisStore.load(path)
    assert restored.selector == selector
    assert restored.storage_breakdown() == store.storage_breakdown()
    x = {key: torch.tensor([[0., 0., 1.]])}
    store.update(x, 1)
    restored.update(x, 1)
    torch.testing.assert_close(covariance(store.history[key]), covariance(restored.history[key]))
    torch.testing.assert_close(next(iter(store.bases.values())), next(iter(restored.bases.values())))
    with pytest.raises(ValueError, match="same module/depth"):
        restored.update({key.replace("ut0", "ut1"): x[key]}, 1)


def test_randomized_history_and_pca_preserve_rank_and_orthogonality():
    torch.manual_seed(8)
    histories = [update_history(None, torch.randn(50, 32), 12) for _ in range(4)]
    q, _ = allocate_joint(histories, 5, "joint_pca")
    assert q.shape == (32, 5)
    torch.testing.assert_close(q.T @ q, torch.eye(5), atol=1e-5, rtol=1e-5)


def test_joint_guard_handles_actual_adamw_updates_and_rejects_bad_basis():
    model = TinyOuro()
    key = "layers.0.self_attn.q_proj"
    q = torch.eye(5)[:, :2]
    guard = HistoricalFullWeightProjection(model, {}, joint_bases={key: q})
    weight = model.model.layers[0].self_attn.q_proj.weight
    target = weight.detach().clone() @ q
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.1)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(2, 3, 5)).square().mean().backward()
        optimizer.step()
        guard.after_step()
        torch.testing.assert_close(weight @ q, target, atol=1e-6, rtol=1e-5)
    assert guard.diagnostics()["basis_representation"] == "joint"
    with pytest.raises(ValueError, match="orthonormal"):
        HistoricalFullWeightProjection(model, {}, joint_bases={key: q * 2})


def test_new_config_validation_and_legacy_protocol_stability():
    base = ["train", "--output-dir", "runs/unused"]
    old = training_config(parse_args(base))
    assert not {"basis_strategy", "history_rank", "basis_svd_method"} & old.keys()
    for extra in (["--basis-strategy", "joint_greedy"],
                  ["--method", "projected", "--basis-strategy", "joint_pca", "--basis-depth-mode", "task"],
                  ["--history-rank", "0"]):
        with pytest.raises(SystemExit):
            parse_args(base + extra)


def test_adaptive_pbs_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PBS_DRY_RUN": "1", "PBS_O_WORKDIR": str(root),
           "RUN_DIR": str(tmp_path / "unused"), "BASIS_STRATEGY": "joint_pca", "SEED": "43"}
    result = subprocess.run(["bash", "pbs/train_adaptive.pbs"], cwd=root, env=env,
                            capture_output=True, text=True, check=True)
    config = json.loads(result.stdout)
    assert config["basis_strategy"] == "joint_pca"
    assert config["basis_depth_mode"] == "all"
    assert config["history_rank"] == config["basis_max_rank"] == 512
    assert config["seed"] == 43
    assert not (tmp_path / "unused").exists()
