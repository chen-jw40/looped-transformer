import json

import pytest
import torch
from torch import nn

from ouro_pipeline.olora import OLoRALinear, ContinualOLoRA
from ouro_pipeline.modeling import final_exit_loss, set_loop_count
from scripts.parameters import parse_args, training_config
from tests.test_osft import tiny_ouro


def test_reference_overlap_is_input_factor_l1_and_norms_are_unsquared():
    module = OLoRALinear(nn.Linear(3, 4, bias=False), rank=2, alpha=4, dropout=0)
    module.old_A = torch.tensor([[1., 0., 0.], [0., 2., 0.]])
    module.old_B = torch.zeros(4, 2)
    with torch.no_grad():
        module.A.copy_(torch.tensor([[1., -1., 0.], [2., 3., 0.]]))
        module.B.zero_()
    overlap, norm = module.penalties()
    assert overlap.item() == 11  # abs([[1,2],[-2,6]]).sum(), not squares
    assert norm.item() == pytest.approx(15**0.5)
    overlap.backward()
    torch.testing.assert_close(module.A.grad, torch.tensor([[1., -2., 0.], [1., 2., 0.]]))
    assert module.old_A.grad is None and not module.base.weight.requires_grad


def test_bf16_autocast_keeps_qkv_dtypes_compatible_with_sdpa():
    model = tiny_ouro()
    model.config._attn_implementation = "sdpa"
    controller = ContinualOLoRA(model, rank=2)
    batch = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 4, 5, 2]])}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = final_exit_loss(model, batch, 4)
    (loss + controller.regularization_loss()).backward()
    assert all(module.B.grad is not None and torch.isfinite(module.B.grad).all()
               for module in controller.modules.values())


def test_freezing_incremental_accumulation_and_export_preserve_learned_updates():
    torch.manual_seed(8)
    module = OLoRALinear(nn.Linear(5, 7), rank=2, alpha=4, dropout=0)
    original = {k: v.clone() for k, v in module.base.state_dict().items()}
    with torch.no_grad():
        module.B.normal_()
    x = torch.randn(3, 5)
    before = module(x).detach()
    module.consolidate()
    torch.testing.assert_close(module(x), before, rtol=0, atol=0)
    old_a, old_b = module.old_A.clone(), module.old_B.clone()
    module.reset_current()
    optimizer = torch.optim.AdamW([module.A, module.B], lr=0.02)
    for _ in range(3):
        optimizer.zero_grad()
        (module(x).square().mean() + 0.5 * module.penalties()[0]).backward()
        optimizer.step()
    assert all(torch.equal(v, original[k]) for k, v in module.base.state_dict().items())
    assert torch.equal(old_a, module.old_A) and torch.equal(old_b, module.old_B)
    second = module(x).detach()
    assert not torch.equal(second, before)
    module.consolidate()
    assert module.old_A.shape == (4, 5) and module.old_B.shape == (7, 4)
    rng = torch.random.get_rng_state().clone()
    dense = module.to_linear()
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(dense(x), second, rtol=2e-5, atol=2e-6)


def test_real_ouro_shared_loops_checkpointing_best_restore_dense_save_and_next_task(tmp_path):
    from ouro_pipeline.early_stopping import TaskEarlyStopping
    from tests.test_early_stopping import Tokenizer
    from transformers import AutoModelForCausalLM
    from unittest.mock import patch

    model = tiny_ouro()
    dense_keys = set(model.state_dict())
    base = {k: p.clone() for k, p in model.named_parameters()}
    batch = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 4, 5, 2]])}
    before = {}
    for loops in (1, 2, 4):
        set_loop_count(model, loops)
        before[loops] = final_exit_loss(model, batch, loops)[1].detach()
    controller = ContinualOLoRA(model, rank=2, dropout=0)
    assert len(controller.modules) == 2  # one q/v pair reused at every loop
    assert all(name.endswith((".A", ".B")) for name, p in model.named_parameters() if p.requires_grad)
    for loops in (1, 2, 4):
        set_loop_count(model, loops)
        torch.testing.assert_close(final_exit_loss(model, batch, loops)[1], before[loops], rtol=0, atol=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
    for _ in range(2):
        optimizer.zero_grad()
        loss, _ = final_exit_loss(model, batch, 4)
        (loss + controller.regularization_loss()).backward()
        optimizer.step()
    (tmp_path / "FOMC").mkdir()
    (tmp_path / "FOMC/eval.json").write_text(json.dumps([{"prompt": "Q", "answer": "A"}]))
    args = parse_args(["train", "--method", "olora", "--output-dir", str(tmp_path), "--data-root", str(tmp_path)])
    stopper = TaskEarlyStopping(model, Tokenizer(), args, "FOMC", 1, None)
    best = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    with patch.object(stopper, "validation_loss", return_value=1):
        stopper.check(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1)
    stopper.restore()
    assert all(torch.equal(best[k], p) for k, p in model.named_parameters() if p.requires_grad)
    model.eval()
    selected = final_exit_loss(model, batch, 4)[1].detach()
    controller.consolidate()
    state = controller.adapter_state()
    assert state["stage"] == 1
    rng = torch.random.get_rng_state().clone()
    with controller.dense_export():
        assert set(model.state_dict()) == dense_keys
        torch.testing.assert_close(final_exit_loss(model, batch, 4)[1], selected, rtol=2e-5, atol=2e-6)
        model.save_pretrained(tmp_path / "dense", safe_serialization=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "dense", trust_remote_code=True,
                                                   attn_implementation="eager")
    torch.testing.assert_close(final_exit_loss(reloaded, batch, 4)[1], selected, rtol=2e-5, atol=2e-6)
    controller.begin_task(2)
    assert all(m.old_A.shape[0] == 2 for m in controller.modules.values())
    assert controller.regularization_loss().item() > 0
    for name, p in model.named_parameters():
        if p.requires_grad or ".old_" in name:
            continue
        original_name = name.replace(".base.", ".")
        assert torch.equal(p, base[original_name])
    with pytest.raises(ValueError):
        controller.begin_task(3)


def test_olora_defaults_all_stage_test_and_rejects_invalid_settings():
    args = parse_args(["train", "--method", "olora", "--output-dir", "runs/test"])
    c = training_config(args)
    assert args.checkpoint_scope == "all" and args.learning_rate == 1e-5
    assert c["olora_target_modules"] == ["q_proj", "v_proj"] and "route_depths" not in c
    for extra in (["--olora-rank", "0"], ["--olora-alpha", "nan"],
                  ["--olora-dropout", "1"], ["--olora-orthogonality-weight", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["train", "--method", "olora", "--output-dir", "runs/test", *extra])


def test_training_adds_regularization_and_reports_ce_separately(tmp_path):
    from ouro_pipeline.training import train_task
    from tests.test_early_stopping import TinyModel, Tokenizer
    from unittest.mock import patch

    args = parse_args(["train", "--method", "olora", "--output-dir", str(tmp_path),
                       "--train-limit", "2", "--gradient-accumulation", "2",
                       "--no-early-stopping", "--epochs", "1", "--max-length", "20", "--learning-rate", "0.1"])
    model = TinyModel()

    class Penalty:
        def regularization_loss(self):
            return (model.weight - 5).square().mean()

    with patch("ouro_pipeline.training.move", side_effect=lambda batch, device: batch):
        report = train_task(model, Tokenizer(), [{"prompt": "Q", "answer": "A"}] * 2,
                            args, "FOMC", regularizer=Penalty())
    assert report["mean_regularization_loss"] == 25
    assert report["mean_loss"] == pytest.approx(float(torch.log(torch.tensor(8.))))
    assert model.weight.sum().item() > 0  # penalty updates logits' common offset
