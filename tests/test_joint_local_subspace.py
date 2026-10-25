import pytest
import torch

from ouro_pipeline.adaptive_subspace import allocate_joint, update_history
from scripts.parameters import parse_args, training_config


def test_local_gain_excludes_benefits_to_other_depths_but_competes_globally():
    eye = torch.eye(3)
    majority = {"basis": eye[:, :2], "eigenvalues": torch.tensor([.6, .4]), "full_trace": 1.}
    minority = {"basis": eye[:, 2:], "eigenvalues": torch.ones(1), "full_trace": 1.}
    histories = [majority] * 3 + [minority]
    joint, _ = allocate_joint(histories, 1, "joint_greedy")
    local, report = allocate_joint(histories, 1, "joint_local")
    # e0 helps three depths, but e2 has the largest source-depth-only gain.
    torch.testing.assert_close(joint @ joint.T, torch.diag(torch.tensor([1., 0., 0.])))
    torch.testing.assert_close(local @ local.T, torch.diag(torch.tensor([0., 0., 1.])))
    assert report["source_counts"] == [0, 0, 0, 1]
    assert report["coverage"] == pytest.approx([0., 0., 0., 1.])
    assert report["joint_coverage"] == pytest.approx(.25)
    assert report["selection_gain_scope"] == "source_depth_only"


def test_local_gains_match_dense_covariance_oracle_after_global_projection():
    torch.manual_seed(103)
    histories = [update_history(None, torch.randn(12, 7), 4, "exact") for _ in range(4)]
    # Exercise normalization without renormalizing retained eigenvalues.
    for history, trace in zip(histories, [1., 2., .8, 1.5]):
        history["full_trace"] = trace
    candidates = [(h["basis"][:, j].double(), d)
                  for d, h in enumerate(histories) for j in range(h["basis"].shape[1])]
    covariances = [(h["basis"].double() * h["eigenvalues"].double()) @ h["basis"].double().T
                   / h["full_trace"] / len(histories) for h in histories]
    oracle = torch.empty(7, 0, dtype=torch.float64)
    chosen = set()
    origins = []
    expected = []
    for _ in range(5):
        choices = []
        for index, (candidate, origin) in enumerate(candidates):
            if index in chosen:
                continue
            v = candidate - oracle @ (oracle.T @ candidate)
            if v.norm().square() <= 1e-6:
                continue
            v = v / v.norm()
            choices.append((float(v @ covariances[origin] @ v), index, origin, v))
        _, index, origin, v = max(choices, key=lambda item: item[0])
        chosen.add(index)
        origins.append(origin)
        oracle = torch.cat([oracle, v[:, None]], dim=1)
        expected.append(oracle @ oracle.T)
    q, report = allocate_joint(histories, 5, "joint_local")
    for rank, projection in enumerate(expected, 1):
        actual = q[:, :rank].double()
        torch.testing.assert_close(actual @ actual.T, projection, atol=1e-5, rtol=1e-5)
    assert report["source_counts"] == [origins.count(d) for d in range(4)]
    expected_coverage = [float(torch.trace(oracle.T @ c @ oracle)) * len(histories) for c in covariances]
    assert report["coverage"] == pytest.approx(expected_coverage, abs=1e-5)


def test_single_depth_local_and_joint_gains_select_the_same_space():
    torch.manual_seed(12)
    history = update_history(None, torch.randn(15, 9), 6, "exact")
    joint, _ = allocate_joint([history], 3, "joint_greedy")
    local, _ = allocate_joint([history], 3, "joint_local")
    torch.testing.assert_close(local @ local.T, joint @ joint.T)


def test_local_selector_is_saved_in_config_and_can_resume():
    args = parse_args(["train", "--method", "projected", "--output-dir", "runs/unused",
                       "--basis-strategy", "joint_local", "--basis-depth-mode", "all",
                       "--history-rank", "128", "--basis-max-rank", "128", "--resume"])
    config = training_config(args)
    assert config["basis_strategy"] == "joint_local"
    assert config["history_rank"] == config["basis_max_rank"] == 128
    assert "basis_balance_epsilon" not in config

