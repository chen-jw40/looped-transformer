import copy
from unittest.mock import patch

import pytest
import torch

from ouro_pipeline.drsp import DRSPBasisStore, DRSPCollector, drsp_rows
from ouro_pipeline.projected_full import HistoricalFullWeightProjection
from tests.test_depth_coupled import Tiny
from scripts.parameters import parse_args, training_config


@pytest.mark.parametrize("strength", [0., .5, 1., 2.])
def test_covariance_formula_and_shared_probe_exact_expectation(strength):
    torch.manual_seed(4)
    gs = torch.randn(4, 2, 5, dtype=torch.float64)
    # All signs: R R.T = I exactly, so the sketched covariance is exact.
    r = torch.tensor([[-1., -1., 1., 1.], [-1., 1., -1., 1.]], dtype=gs.dtype) / 2
    zs = gs.transpose(1, 2) @ r
    rows = drsp_rows(zs, strength)
    mean = gs.mean(0)
    h = gs - mean
    expected = mean.T @ mean + strength / 4 * sum(g.T @ g for g in h)
    torch.testing.assert_close(rows.T @ rows, expected)
    torch.testing.assert_close(expected, (1-strength)*mean.T@mean + strength/4*sum(g.T@g for g in gs))
    same = gs[:1].expand(4, -1, -1).transpose(1, 2) @ r
    torch.testing.assert_close(drsp_rows(same, strength).T @ drsp_rows(same, strength), gs[0].T@gs[0])
    one = drsp_rows(zs[:1], strength)
    torch.testing.assert_close(one.T @ one, zs[0] @ zs[0].T)


def test_cancellation_is_retained_only_when_lambda_positive():
    z = torch.tensor([[[1.], [0.]], [[-1.], [0.]]])
    assert drsp_rows(z, 0).square().sum() == 0
    torch.testing.assert_close(drsp_rows(z, .5).T @ drsp_rows(z, .5), torch.diag(torch.tensor([.5, 0.])))


@pytest.mark.parametrize("checkpointed", [False, True])
def test_normal_backward_matches_call_gradients_and_does_not_change_training_rng(checkpointed):
    torch.manual_seed(14)
    model = Tiny(checkpointed)
    reference = copy.deepcopy(model)
    x = torch.randn(1, 5, 5, requires_grad=True)
    reference(x).square().mean().div(16).backward()
    observer = DRSPCollector(model, 4, samples=2, width=3, seed=8)
    observer.configure(2)
    observations = []
    def observe(_module, inputs, output):
        if observer.recording:
            output.retain_grad()
            observations.append((inputs[0].detach().clone(), output))
    handle = model.model.layers[0].self_attn.o_proj.register_forward_hook(observe)
    rng = torch.get_rng_state().clone()
    batch = {"input_ids": torch.ones(1, 5, dtype=torch.long), "labels": torch.ones(1, 5, dtype=torch.long)}
    observer.begin(0, batch, 16)
    loss = model(x).square().mean()
    key = "layers.0.self_attn.o_proj"
    r = observer.probes[key].clone()
    observer.before_backward()
    (loss / 16).backward()
    gs = torch.stack([out.grad.reshape(-1, 5).T @ inp.reshape(-1, 5) * 16 for inp, out in observations])
    assert len(observations) == 4
    torch.testing.assert_close(gs.sum(0), model.model.layers[0].self_attn.o_proj.weight.grad * 16)
    observer.finish()
    torch.testing.assert_close(observer.tensors()[key], drsp_rows(gs.transpose(1, 2) @ r, .5), atol=1e-7, rtol=2e-5)
    assert torch.equal(rng, torch.get_rng_state())
    for name, p in model.named_parameters():
        ref = dict(reference.named_parameters())[name]
        if ref.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, ref.grad)
    assert observer.tensors()["early_exit_gate"].square().sum() == 0
    assert observer.diagnostics()["extra_backward_passes"] == 0
    observer.close()
    handle.remove()
    assert not observer.tensor_handles and not observer.handles


def test_history_keeps_strength_reselects_at_capacity_and_roundtrips(tmp_path):
    store = DRSPBasisStore(1, "exact")
    store.update({"x": torch.tensor([[2., 0.]])})
    store.update({"x": torch.tensor([[0., 3.]])})
    # Spectral mass 9 wins over old 4; no saturation or equal unit-basis merge.
    torch.testing.assert_close(store.bases["x"].abs(), torch.tensor([[0.], [1.]]))
    torch.testing.assert_close(store.history["x"]["eigenvalues"], torch.tensor([9.]))
    assert store.history["x"]["full_trace"] == 13
    store.save(tmp_path / "basis.pt")
    restored = DRSPBasisStore.load(tmp_path / "basis.pt")
    assert restored.storage_bytes() == store.storage_bytes()
    torch.testing.assert_close(restored.bases["x"], store.bases["x"])
    exact = DRSPBasisStore(2, "exact")
    exact.update({"x": torch.tensor([[2., 0.]])})
    exact.update({"x": torch.tensor([[0., 3.]])})
    h = exact.history["x"]
    torch.testing.assert_close((h["basis"] * h["eigenvalues"]) @ h["basis"].T, torch.diag(torch.tensor([4., 9.])))
    zero = DRSPBasisStore(2, "randomized")
    zero.update({"x": torch.zeros(20, 12)})
    assert zero.bases["x"].shape == (12, 0)


def test_history_repairs_svd_orthogonality_without_changing_retained_covariance():
    u = torch.tensor([[1.0003, .0004], [0., .9997], [0., 0.]])
    s = torch.tensor([3., 2.])
    covariance = (u * s.square()) @ u.T
    store = DRSPBasisStore(2)
    with patch("ouro_pipeline.drsp._svd", return_value=(u, s)):
        store.update({"x": torch.ones(2, 3)})
    q, values = store.bases["x"], store.history["x"]["eigenvalues"]
    torch.testing.assert_close(q.T @ q, torch.eye(2), atol=2e-7, rtol=2e-7)
    torch.testing.assert_close((q * values) @ q.T, covariance, atol=2e-6, rtol=2e-6)


def test_real_ouro_two_tasks_normal_backward_checkpointing_and_adam_projection(tmp_path):
    from tests.test_osft import tiny_ouro
    from tests.test_early_stopping import Tokenizer
    from ouro_pipeline import full_training, training
    model = tiny_ouro()
    args = parse_args(["train", "--method", "projected", "--basis-statistic", "drsp",
                      "--output-dir", str(tmp_path / "run"), "--tasks", "FOMC", "ScienceQA",
                      "--train-limit", "2", "--stat-samples", "2", "--basis-max-rank", "2",
                      "--drsp-sketch-width", "2", "--gradient-accumulation", "2", "--no-early-stopping",
                      "--max-length", "20"])
    # Existing orchestration is CUDA-only; run its real tiny Ouro loop on CPU.
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "Q", "answer": "AA"}]
    fork_rng = torch.random.fork_rng
    with patch.object(full_training, "load_ouro", return_value=model), \
         patch.object(full_training, "load_tokenizer", return_value=Tokenizer()), \
         patch.object(full_training, "read_split", return_value=records), \
         patch.object(full_training.torch.cuda, "is_available", return_value=True), \
         patch.object(full_training.torch.cuda, "is_bf16_supported", return_value=True), \
         patch.object(full_training.torch.cuda, "reset_peak_memory_stats"), \
         patch.object(full_training.torch.cuda, "synchronize"), \
         patch.object(full_training.torch.cuda, "max_memory_allocated", return_value=0), \
         patch.object(full_training.torch.cuda, "max_memory_reserved", return_value=0), \
         patch.object(full_training.torch.cuda, "current_device", return_value=0), \
         patch.object(full_training.torch.random, "fork_rng", side_effect=lambda **kw: fork_rng(devices=[])), \
         patch.object(training, "move", side_effect=lambda b, _: b):
        with patch.object(Tokenizer, "save_pretrained", create=True):
            # Provenance hashes read actual data files, which exist in this repo.
            full_training.run(args)
    import json
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["training_status"] == "complete"
    assert len(summary["stages"]) == 2
    second = summary["training"][1]
    assert second["drsp_statistics"]["samples"] == 2
    assert second["projection_diagnostics"]["protected_linear_weights"] == 8
    assert second["projection_diagnostics"]["max_WU_history_scaled_error"] < 1e-6
    for stage in summary["stages"]:
        state = DRSPBasisStore.load(stage["basis"])
        assert all(q.shape[1] <= 2 for q in state.bases.values())
        assert state.bases["early_exit_gate"].shape[1] == 0


def test_cli_legacy_and_invalid_drsp():
    base = ["train", "--output-dir", "runs/example"]
    assert "drsp_lambda" not in training_config(parse_args(base))
    good = base + ["--method", "projected", "--basis-statistic", "drsp"]
    assert training_config(parse_args(good))["drsp_lambda"] == .5
    assert parse_args(good + ["--micro-batch-size", "8", "--resume"]).micro_batch_size == 8
    for extra in (["--drsp-lambda", "nan"],
                  ["--drsp-lambda", "-1"], ["--drsp-sketch-width", "0"], ["--basis-strategy", "joint_pca"]):
        with pytest.raises(SystemExit):
            parse_args(good + extra)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_batch8_sketch_equals_individual_ce_with_unequal_answer_lengths(checkpointed):
    torch.manual_seed(17)
    model = Tiny(checkpointed)
    reference = copy.deepcopy(model)
    untouched = copy.deepcopy(model)
    x = torch.randn(8, 10, 5)
    labels = torch.randint(0, 7, (8, 10))
    for i in range(8):
        labels[i, :i+1] = -100
    def ce(m, inputs, targets):
        return torch.nn.functional.cross_entropy(m(inputs)[:, :-1].reshape(-1, 7), targets[:, 1:].reshape(-1))
    ce(untouched, x, labels).div(16).backward()
    observers = []
    for m, inputs, targets in [(model, x, labels), (reference, x[:1], labels[:1])]:
        observer = DRSPCollector(m, 4, samples=1, width=3, seed=8)
        observers.append(observer)
        observer.configure(1)
        observer.begin(0, {"input_ids": torch.ones_like(targets), "labels": targets}, 16)
        loss = ce(m, inputs, targets)
        observer.before_backward()
        loss.div(16).backward()
        observer.finish()
    for key, tensor in observers[0].tensors().items():
        torch.testing.assert_close(tensor, observers[1].tensors()[key], atol=2e-7, rtol=3e-5)
    for p, ref in zip(model.parameters(), untouched.parameters()):
        if ref.grad is not None:
            torch.testing.assert_close(p.grad, ref.grad, rtol=0, atol=0)
    for observer in observers:
        observer.close()


def test_batch8_drsp_recovery_preserves_sketches_rng_and_weights(tmp_path):
    import os
    from tests.test_osft import tiny_ouro
    from tests.test_early_stopping import Tokenizer
    from ouro_pipeline import training, recovery
    torch.manual_seed(45)
    initial = tiny_ouro()
    initial.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    records = [{"prompt": "Q" * (i % 3 + 1), "answer": "A" * (i % 4 + 1)} for i in range(23)]
    args = parse_args(["train", "--output-dir", str(tmp_path), "--method", "projected",
                      "--basis-statistic", "drsp", "--tasks", "FOMC", "--epochs", "3",
                      "--micro-batch-size", "8", "--max-length", "24", "--gradient-accumulation", "2"])
    def observer(m):
        return DRSPCollector(m, 4, samples=7, width=2, seed=8)
    ref = copy.deepcopy(initial)
    expected_observer = observer(ref)
    with patch.object(training, "move", side_effect=lambda b, _: b), patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "0"}):
        expected = training.train_task(ref, Tokenizer(), records, args, "FOMC", gradient_observer=expected_observer)
    with patch.object(training, "move", side_effect=lambda b, _: b), patch.dict(os.environ, {"OURO_RECOVERY_ENABLED": "1"}):
        for step in (1, 2, 3):
            model = copy.deepcopy(initial)
            obs = observer(model)
            with patch.object(recovery, "deadline_reached", return_value=True), pytest.raises(recovery.TrainingPaused):
                training.train_task(model, Tokenizer(), records, args, "FOMC", gradient_observer=obs)
            obs.close()
            assert torch.load(recovery.checkpoint_path(args, "FOMC"), weights_only=False)["progress"]["optimizer_steps"] == step
        model = copy.deepcopy(initial)
        obs = observer(model)
        with patch.object(recovery, "deadline_reached", return_value=False):
            actual = training.train_task(model, Tokenizer(), records, args, "FOMC", gradient_observer=obs)
    assert actual == expected
    for name, tensor in ref.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    for key, tensor in expected_observer.tensors().items():
        torch.testing.assert_close(obs.tensors()[key], tensor, rtol=0, atol=0)
    a, b = obs.diagnostics(), expected_observer.diagnostics()
    a.pop("hook_cpu_wall_seconds"); b.pop("hook_cpu_wall_seconds")
    assert a == b
    obs.close()
    expected_observer.close()


def test_allocation_maps_exactly_three_distinct_nodes(tmp_path):
    from scripts.drsp_study import allocation_nodes
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node1\nnode2\nnode3\n")
    with patch.dict("os.environ", {"PBS_NODEFILE": str(nodefile)}):
        assert allocation_nodes() == ["node1", "node2", "node3"]
        nodefile.write_text("node1\nnode1\nnode2\n")
        with pytest.raises(RuntimeError):
            allocation_nodes()
