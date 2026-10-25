import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ouro_pipeline.depth_coupled import DepthCoupledCollector, calibration_mode
from ouro_pipeline.projected_full import HistoricalFullWeightProjection
from ouro_pipeline.subspace import BasisStore
from scripts.parameters import parse_args, training_config


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(5, 5, bias=False)

    def forward(self, x, current_ut):
        return x + self.self_attn.o_proj(x).tanh()


class Tiny(nn.Module):
    def __init__(self, checkpointed=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Decoder()])
        self.model.norm = nn.LayerNorm(5)
        self.model.early_exit_gate = nn.Linear(5, 1)
        self.lm_head = nn.Linear(5, 7)
        self.checkpointed = checkpointed

    def forward(self, x):
        for depth in range(4):
            if self.checkpointed:
                x = checkpoint(self.model.layers[0], x, current_ut=depth, use_reentrant=False)
            else:
                x = self.model.layers[0](x, current_ut=depth)
            x = self.model.norm(x)
            self.model.early_exit_gate(x)
        return self.lm_head(x)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_sketch_matches_explicit_per_position_gradient_and_preserves_training(checkpointed, independent):
    torch.manual_seed(14)
    model = Tiny(checkpointed)
    reference = copy.deepcopy(model)
    x = torch.randn(1, 5, 5, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    reference(x).square().sum().backward()
    observations = []
    with DepthCoupledCollector(model, 4, 6, 3, seed=8, independent=independent) as collector:
        def observe(_module, inputs, output):
            if collector.recording:
                output.retain_grad()
                observations.append((inputs[0].detach().clone(), output))
        h = model.model.layers[0].self_attn.o_proj.register_forward_hook(observe)
        rng = torch.get_rng_state().clone()
        collector.start(mask)
        loss = model(x).square().sum()
        collector.backward(loss)
        key = "layers.0.self_attn.o_proj"
        indices = torch.tensor([0, 1, 3])
        expected = torch.zeros(3, 5)
        for depth, (inputs, output) in enumerate(observations):
            b = output.grad[0, indices]
            r = collector.probes[(key, depth if independent else 0)]
            expected += (b * r).sum(-1, keepdim=True) * inputs[0, indices]
        assert len(observations) == 4
        collector.finish()
        torch.testing.assert_close(collector.tensors()[key], expected)
        assert torch.equal(rng, torch.get_rng_state())
        assert set(collector.tensors()) == {key, "lm_head", "early_exit_gate"}
        h.remove()
        for name, p in model.named_parameters():
            other = dict(reference.named_parameters())[name]
            if other.grad is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, other.grad)
        collector.start(mask)
        collector.backward(model(x).square().sum())
        collector.finish()
        assert all(v.shape == (6, 5) for v in collector.tensors().values())
        collector.start(mask)
        collector.backward(model(x).square().sum())
        collector.finish()
        assert all(v.shape == (6, 5) for v in collector.tensors().values())
    assert not collector.handles and not collector.tensor_handles


def test_same_probe_preserves_cross_terms_exactly_in_expectation():
    torch.manual_seed(2)
    x, b = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
    a = b.T @ x
    probes = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=x.dtype)
    z = probes @ a
    torch.testing.assert_close(z.T @ z / 4, a.T @ a)
    diagonal = sum(b[k].square().sum() * torch.outer(x[k], x[k]) for k in range(4))
    assert not torch.allclose(a.T @ a, diagonal)


def test_sketch_basis_is_enforced_after_adamw():
    torch.manual_seed(4)
    model = Tiny(True)
    with DepthCoupledCollector(model, 4, 4, 4) as collector:
        collector.start(torch.ones(1, 4, dtype=torch.long))
        collector.backward(model(torch.randn(1, 4, 5)).square().sum())
        collector.finish()
        store = BasisStore()
        store.update(collector.tensors(), max_rank=2, residual_tolerance=.05)
    model.zero_grad(set_to_none=True)
    guard = HistoricalFullWeightProjection(model, {}, joint_bases=store.bases)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
    for _ in range(3):
        model(torch.randn(1, 4, 5)).square().sum().backward()
        optimizer.step()
        guard.after_step()
        optimizer.zero_grad(set_to_none=True)
    for _, weight, q, target, *_ in guard.constraints:
        torch.testing.assert_close(weight @ q, target, atol=1e-6, rtol=1e-5)


def test_modes_restored_on_failure():
    model = Tiny().eval()
    model.dropout = nn.Dropout(.5)
    model.model.layers[0].attention_dropout = .1
    modes = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError), calibration_mode(model):
        assert model.training and not model.dropout.training
        assert model.model.layers[0].attention_dropout == 0
        raise RuntimeError("failure")
    assert [m.training for m in model.modules()] == modes
    assert model.model.layers[0].attention_dropout == .1


def test_cli_rejects_incompatible_protocols_and_preserves_legacy_config():
    base = ["train", "--output-dir", "runs/example"]
    assert "basis_statistic" not in training_config(parse_args(base))
    valid = base + ["--method", "projected", "--basis-statistic", "depth_coupled"]
    assert parse_args(valid).basis_depth_mode == "all"
    for extra in (["--method", "full"], ["--basis-depth-mode", "last"],
                  ["--basis-strategy", "joint_pca"], ["--route-depths", "0"]):
        with pytest.raises(SystemExit):
            parse_args(valid + extra)


def test_real_ouro_checkpoint_calibration_preserves_weights_rng_and_modes(tmp_path):
    from ouro_pipeline.depth_coupled import collect_coupled_history
    from tests.test_osft import tiny_ouro
    from tests.test_early_stopping import Tokenizer

    model = tiny_ouro()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = parse_args(["train", "--method", "projected", "--basis-statistic", "depth_coupled",
        "--output-dir", str(tmp_path), "--max-length", "20", "--stat-samples", "2"])
    rng = torch.get_rng_state().clone()
    tensors, n, diagnostics = collect_coupled_history(model, Tokenizer(),
        [{"prompt": "Q", "answer": "A"}, {"prompt": "Q", "answer": "AA"}], args, "FOMC", 1)
    assert n == 2 and diagnostics["extra_calibration_backward_passes"] == 2
    assert tensors and all(torch.isfinite(v).all() for v in tensors.values())
    assert sum(v.square().sum() for k, v in tensors.items() if k.startswith("layers.")) > 0
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("statistic", ["depth_coupled", "depth_diagonal", "activation"])
def test_container_precision_override_is_explicit_and_scoped(statistic):
    from scripts.runtime import container_command
    argv = ["train", "--method", "projected", "--basis-statistic", statistic]
    with patch("scripts.runtime.shutil.which", return_value="/usr/bin/apptainer"), \
         patch("scripts.runtime.Path.is_file", return_value=True), \
         patch.dict("os.environ", {}, clear=True):
        _, env = container_command("training", argv)
    for key in ("APPTAINERENV_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "APPTAINERENV_NVIDIA_TF32_OVERRIDE"):
        if statistic == "activation":
            assert key not in env
        else:
            assert env[key] == "0"
