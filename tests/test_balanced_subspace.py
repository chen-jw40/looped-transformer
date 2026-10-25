import pytest
import torch

from ouro_pipeline.adaptive_subspace import allocate_joint, update_history
from scripts.parameters import parse_args, training_config
from tests.test_adaptive_subspace import covariance


def test_log_coverage_protects_the_underrepresented_depth():
    eye = torch.eye(3)
    majority = {"basis": eye[:, :2], "eigenvalues": torch.tensor([.6, .4]), "full_trace": 1.}
    minority = {"basis": eye[:, 2:], "eigenvalues": torch.ones(1), "full_trace": 1.}
    histories = [majority] * 3 + [minority]
    old, _ = allocate_joint(histories, 2)
    q, details = allocate_joint(histories, 2, "joint_balanced")
    torch.testing.assert_close(old @ old.T, torch.diag(torch.tensor([1., 1., 0.])))
    torch.testing.assert_close(q @ q.T, torch.diag(torch.tensor([1., 0., 1.])))
    assert details["coverage"] == pytest.approx([.6, .6, .6, 1.])
    assert details["depth_weights"] == pytest.approx([.25] * 4)


def test_balanced_gains_match_direct_covariance_objective():
    torch.manual_seed(19)
    histories = [update_history(None, torch.randn(9, 6), 4, "exact") for _ in range(4)]
    candidates = torch.cat([h["basis"].double() for h in histories], dim=1)
    matrices = [covariance(h).double() for h in histories]
    oracle = torch.empty(6, 0, dtype=torch.float64)
    for _ in range(3):
        choices = []
        for candidate in candidates.T:
            v = candidate - oracle @ (oracle.T @ candidate)
            if v.norm().square() <= 1e-6:
                continue
            next_q = torch.cat([oracle, (v / v.norm())[:, None]], dim=1)
            objective = sum(torch.log(.01 + torch.trace(next_q.T @ c @ next_q)) for c in matrices)
            choices.append((float(objective), next_q))
        oracle = max(choices, key=lambda item: item[0])[1]
    q, report = allocate_joint(histories, 3, "joint_balanced")
    torch.testing.assert_close((q @ q.T).double(), oracle @ oracle.T, atol=1e-5, rtol=1e-5)
    expected = sum(torch.log(.01 + torch.trace(oracle.T @ c @ oracle)) for c in matrices) / 4
    assert report["balanced_objective"] == pytest.approx(float(expected), abs=1e-6)


def test_balanced_configuration_is_explicit_and_legacy_config_unchanged():
    base = ["train", "--method", "projected", "--output-dir", "runs/unused"]
    assert "basis_balance_epsilon" not in training_config(parse_args(base))
    for strategy in ["joint_balanced"]:
        c = training_config(parse_args(base + ["--basis-strategy", strategy]))
        assert c["basis_balance_epsilon"] == .01
    for epsilon in ["0", "nan", "1.1"]:
        with pytest.raises(SystemExit):
            parse_args(base + ["--basis-balance-epsilon", epsilon])
