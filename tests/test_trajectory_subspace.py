from contextlib import nullcontext
import json
from unittest.mock import patch

import pytest
import torch

from ouro_pipeline.subspace import update_basis
from ouro_pipeline.trajectory_subspace import select_trajectory, TrajectoryBasisStore
from scripts.parameters import parse_args


def _paired_example():
    last = torch.diag(torch.tensor([1., .22, 0.]))
    change = torch.zeros_like(last)
    change[0, 1] = 3.
    return [last - (3 - k) * change for k in range(4)]


def test_trajectory_uses_available_last_tolerance_and_preserves_rank():
    torch.manual_seed(4)
    xs = _paired_example()
    q, report = select_trajectory(xs, None, 1, .05, method="exact")
    assert q.shape == (3, 1)
    torch.testing.assert_close(q.T @ q, torch.eye(1))
    assert report["last_residual_fraction"] <= .0500001
    assert report["trajectory_coverage_after"] > report["trajectory_coverage_before"] + .001
    assert report["selection"] == "trajectory"
    assert report["floor_satisfied"]


def test_strict_reference_floor_and_head_keep_last():
    xs = _paired_example()
    expected, _ = update_basis(xs[-1], None, 1, .05)
    strict, report = select_trajectory(xs, None, 1, .05, floor="reference", method="exact")
    assert report["trajectory_coverage_after"] < 1e-6
    head, head_report = select_trajectory(xs, None, 1, .05, head=True)
    torch.testing.assert_close(head @ head.T, expected @ expected.T, atol=1e-6, rtol=1e-6)
    assert head_report["selection"] == "head_last"


def test_old_space_is_permanent_and_saturation_cannot_replace_it():
    old = torch.tensor([[0.], [0.], [1.]])
    q, report = select_trajectory(_paired_example(), old, 2, .05, method="exact")
    torch.testing.assert_close(q @ (q.T @ old), old)
    reference, _ = update_basis(_paired_example()[-1], old, 2, .05)
    assert q.shape == reference.shape
    saturated, details = select_trajectory(_paired_example(), q, 2, .05, method="exact")
    assert details["selection"] == "no_new_capacity"
    torch.testing.assert_close(saturated @ saturated.T, q @ q.T)


def test_infeasible_tolerance_uses_reference_floor_without_inventing_capacity():
    xs = [torch.eye(4) * scale for scale in (1., 2., 3., 4.)]
    q, report = select_trajectory(xs, None, 1, .05, method="exact")
    assert q.shape == (4, 1)
    assert report["last_residual_limit"] == pytest.approx(.75)
    assert report["last_residual_fraction"] <= .7500001


@pytest.mark.parametrize("rank", [64, 128, 512])
def test_requested_rank_caps_on_finite_matrices(rank):
    # A nonzero diagonal plus zero depth differences exercises real dimensions
    # without running an expensive full model in a unit test.
    xs = [torch.eye(rank + 2)] * 4
    q, report = select_trajectory(xs, None, rank, 0., method="exact")
    assert q.shape == (rank + 2, rank)
    assert report["selection"] == "zero_trajectory"
    assert report["new_rank"] == report["reference_rank"] == rank


def test_store_roundtrip_pairing_validation_and_atomic_update(tmp_path):
    values = {f"layers.0.self_attn.o_proj@ut{k}": x for k, x in enumerate(_paired_example())}
    values.update({f"lm_head@ut{k}": x for k, x in enumerate(_paired_example())})
    store = TrajectoryBasisStore("exact")
    reports = store.update(values, 1, .05)
    assert {r["layer"] for r in reports} == {"layers.0.self_attn.o_proj", "lm_head"}
    path = tmp_path / "basis.pt"
    store.save(path)
    restored = TrajectoryBasisStore.load(path)
    for key in store.bases:
        torch.testing.assert_close(restored.bases[key], store.bases[key])
    assert restored.storage_bytes() == store.storage_bytes()
    previous = store.bases
    with pytest.raises(ValueError, match="position-paired"):
        store.update({**values, "lm_head@ut2": torch.ones(2, 3)}, 1, .05)
    assert store.bases is previous
    with pytest.raises(ValueError, match="missing trajectory"):
        TrajectoryBasisStore().update({k: v for k, v in values.items() if not k.endswith("@ut1")}, 1, .05)


def test_cli_rejects_wrong_trajectory_protocols():
    base = ["train", "--output-dir", "runs/unused", "--basis-strategy", "last_trajectory"]
    for extra in (["--method", "full"], ["--method", "projected", "--basis-depth-mode", "last"],
                  ["--method", "projected", "--loops", "1"],
                  ["--method", "projected", "--basis-statistic", "depth_coupled"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *extra])


def test_real_ouro_training_trajectory_collection_projection_and_export(tmp_path):
    from ouro_pipeline import full_training
    from ouro_pipeline.projected_full import iter_full_linears
    from tests.test_early_stopping import Tokenizer
    from tests.test_osft import tiny_ouro

    torch.manual_seed(42)
    model, tokenizer = tiny_ouro(), Tokenizer()
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "QQ", "answer": "AA"}]
    for task in ("FOMC", "ScienceQA"):
        directory = tmp_path / "data" / task
        directory.mkdir(parents=True)
        for split in ("train", "eval", "test"):
            (directory / f"{split}.json").write_text(json.dumps(records))
    args = parse_args(["train", "--method", "projected", "--output-dir", str(tmp_path / "run"),
        "--data-root", str(tmp_path / "data"), "--tasks", "FOMC", "ScienceQA", "--loops", "4",
        "--basis-strategy", "last_trajectory", "--no-early-stopping", "--gradient-accumulation", "1",
        "--train-limit", "2", "--stat-samples", "2", "--stat-vectors-per-layer", "4",
        "--stat-vectors-per-call", "2", "--basis-max-rank", "2", "--basis-svd-method", "exact"])
    cpu_rng = torch.random.fork_rng
    with patch.object(full_training, "load_ouro", return_value=model), \
         patch.object(full_training, "load_tokenizer", return_value=tokenizer), \
         patch.object(tokenizer, "save_pretrained", create=True), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.autocast", side_effect=lambda **kwargs: nullcontext()), \
         patch("torch.random.fork_rng", side_effect=lambda **kwargs: cpu_rng(devices=[])), \
         patch("ouro_pipeline.training.move", side_effect=lambda batch, device: batch):
        full_training.run(args)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["training_status"] == "complete"
    assert summary["projection_protocol"]["extra_calibration_backward_passes"] == 0
    stores = [TrajectoryBasisStore.load(stage["basis"]) for stage in summary["stages"]]
    expected = {key for key, _ in iter_full_linears(model)}
    assert set(stores[0].bases) == set(stores[1].bases) == expected
    for stage in summary["stages"]:
        assert stage["basis_source_depths"] == [0, 1, 2, 3]
        assert len(stage["sampled_vectors_by_depth"]) == 4 * len(expected)
        assert all(r["floor_satisfied"] and r["reference_rank"] == r["new_rank"]
                   for r in stage["basis_updates"])
    for key in expected:
        old, new = stores[0].bases[key], stores[1].bases[key]
        torch.testing.assert_close(new @ (new.T @ old), old, atol=1e-5, rtol=1e-5)
    diag = summary["training"][1]["projection_diagnostics"]
    assert diag["protected_linear_weights"] == len(expected)
    assert diag["max_WU_history_scaled_error"] < 1e-5
    assert all(rank <= 2 for rank in diag["history_union_ranks"].values())
