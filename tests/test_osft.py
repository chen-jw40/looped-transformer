import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from ouro_pipeline.modeling import final_exit_loss, set_loop_count
from ouro_pipeline.osft import OSFTLinear, StageOSFT, retention_ratio
from scripts.parameters import REPO_ROOT, parse_args, training_config


@pytest.mark.parametrize("shape", [(7, 7), (11, 5), (5, 11)])
def test_rectangular_reconstruction_and_dense_export_preserve_forward(shape):
    torch.manual_seed(31)
    linear = nn.Linear(shape[1], shape[0])
    wrapper = OSFTLinear(linear, 2)
    x = torch.randn(2, 3, shape[1])
    torch.testing.assert_close(wrapper(x), linear(x), rtol=1e-5, atol=1e-6)
    state = torch.random.get_rng_state().clone()
    dense = wrapper.to_linear()
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(wrapper(x), dense(x), rtol=0, atol=0)
    assert set(dense.state_dict()) == {"weight", "bias"}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(wrapper(x), dense(x), rtol=0, atol=0)


def test_protected_left_and_right_actions_survive_adamw_but_model_can_learn():
    torch.manual_seed(42)
    wrapper = OSFTLinear(nn.Linear(9, 7, bias=False), 3)
    start = wrapper.dense_weight().detach().clone()
    frozen = {k: v.clone() for k, v in wrapper.named_buffers()}
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=0.05, weight_decay=0.1)
    for _ in range(8):
        optimizer.zero_grad()
        wrapper(torch.randn(3, 9)).square().mean().backward()
        wrapper.project_gradients()
        optimizer.step()
        wrapper.project_parameters()
        current = wrapper.dense_weight().detach()
        torch.testing.assert_close(wrapper.U_high.T @ current, wrapper.U_high.T @ start, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(current @ wrapper.V_high.T, start @ wrapper.V_high.T, rtol=1e-5, atol=2e-6)
    assert not torch.allclose(start, current)
    assert max(wrapper.orthogonality_errors()) < 1e-5
    assert all(torch.equal(value, frozen[name]) for name, value in wrapper.named_buffers())


def tiny_ouro():
    # Exercise the actual patched remote code without allocating pretrained weights.
    path = REPO_ROOT / "models/Ouro-1.4B"
    if not (path / "modeling_ouro.py").exists():
        pytest.skip("Local Ouro remote code unavailable")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.hidden_size, config.intermediate_size = 16, 24
    config.num_hidden_layers, config.layer_types = 1, ["full_attention"]
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 2, 2, 8
    config.vocab_size, config.max_position_embeddings = 16, 64
    config._attn_implementation = "eager"
    config.use_cache = False
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True).float()


def test_real_ouro_shared_loops_checkpointing_restore_export_and_next_stage():
    torch.manual_seed(7)
    model = tiny_ouro()
    batch = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 4, 5, 2]])}
    dense_keys = set(model.state_dict())
    before = {k: final_exit_loss(model, batch, k)[1].detach() for k in (1, 2, 4)
              if set_loop_count(model, k)}
    guard = StageOSFT(model, 2, 8)
    assert len(guard.modules) == 7  # Physical targets, not 4 x 7 virtual layers.
    for loops in (1, 2, 4):
        set_loop_count(model, loops)
        torch.testing.assert_close(final_exit_loss(model, batch, loops)[1], before[loops], rtol=2e-5, atol=2e-6)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(2):
        loss, _ = final_exit_loss(model, batch, 4)
        loss.backward()
        guard.before_step()
        optimizer.step()
        guard.after_step()
        optimizer.zero_grad(set_to_none=True)
    # The early stopper stores/restores precisely the trainable parameters.
    from ouro_pipeline.early_stopping import TaskEarlyStopping
    from tests.test_early_stopping import Tokenizer
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "FOMC").mkdir()
        (root / "FOMC/eval.json").write_text(json.dumps([{"prompt": "Q", "answer": "A"}]))
        args = parse_args(["train", "--method", "osft", "--output-dir", str(root), "--data-root", str(root)])
        stopper = TaskEarlyStopping(model, Tokenizer(), args, "FOMC", 1, None)
        best = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        with patch.object(stopper, "validation_loss", return_value=1.0):
            stopper.check(1)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.1)
        stopper.restore()
        assert all(torch.equal(p, best[k]) for k, p in model.named_parameters() if p.requires_grad)
        diagnostics = guard.diagnostics()
        assert diagnostics["optimizer_steps_projected"] == 2
        model.eval()
        with torch.no_grad():
            logits = final_exit_loss(model, batch, 4)[1]
        guard.materialize()
        assert set(model.state_dict()) == dense_keys
        torch.testing.assert_close(final_exit_loss(model, batch, 4)[1], logits, rtol=0, atol=0)
        model.save_pretrained(root / "dense", safe_serialization=True)
        reloaded = AutoModelForCausalLM.from_pretrained(root / "dense", trust_remote_code=True,
                                                       attn_implementation="eager")
        torch.testing.assert_close(final_exit_loss(reloaded, batch, 4)[1], logits, rtol=0, atol=0)
    next_guard = StageOSFT(model, 3, 8)
    assert next_guard.ratio == 0.25
    assert all(item["protected_rank"] == item["matrix_rank"] // 4 for item in next_guard.ranks.values())
    next_guard.materialize()


def test_stage_one_is_exact_full_training_and_config_has_no_depth_routes():
    model = tiny_ouro()
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    guard = StageOSFT(model, 1, 8)
    assert guard.modules == {}
    assert all(torch.equal(p, before[k]) for k, p in model.named_parameters())
    args = parse_args(["train", "--method", "osft", "--output-dir", "runs/test"])
    config = training_config(args)
    assert args.learning_rate == 1e-5
    assert config["osft_retention_ratios"] == [i / 8 for i in range(8)]
    assert "route_depths" not in config and "depth_schedule" not in config and "projection" not in config
    assert [retention_ratio(i, 8) for i in range(1, 9)] == config["osft_retention_ratios"]
    with pytest.raises(ValueError):
        retention_ratio(9, 8)


def test_nonfinite_orthogonality_fails_instead_of_being_reported_as_success():
    model = tiny_ouro()
    guard = StageOSFT(model, 2, 8)
    with torch.no_grad():
        next(iter(guard.modules.values())).V_low.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        guard.diagnostics()
