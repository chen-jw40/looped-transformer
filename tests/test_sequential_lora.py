import copy

import pytest
import torch

from ouro_pipeline.lora import ContinualLoRA
from ouro_pipeline.olora import ContinualOLoRA
from scripts.parameters import parse_args, training_config
from tests.test_osft import tiny_ouro


def test_sequential_lora_keeps_one_adapter_and_exports_same_logits():
    model = tiny_ouro()
    base = {name: p.clone() for name, p in model.named_parameters()}
    controller = ContinualLoRA(model, rank=2, alpha=4, dropout=0)
    batch = dict(input_ids=torch.tensor([[1, 3, 4, 2]]), use_cache=False, exit_at_step=3)
    for stage in (1, 2):
        controller.begin_task(stage)
        with torch.no_grad():
            for module in controller.modules.values():
                module.B.add_(.01)
        before = model(**batch).logits.detach()
        factors = {name: (m.A.clone(), m.B.clone()) for name, m in controller.modules.items()}
        controller.consolidate()
        for name, module in controller.modules.items():
            assert module.old_A.shape[0] == 0
            torch.testing.assert_close(module.A, factors[name][0])
            torch.testing.assert_close(module.B, factors[name][1])
            assert module.A.requires_grad and module.B.requires_grad
        with controller.dense_export():
            torch.testing.assert_close(model(**batch).logits, before, atol=2e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        if name.endswith((".A", ".B")):
            continue
        original = name.replace(".base.", ".")
        torch.testing.assert_close(parameter, base[original], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["lora", "olora"])
def test_adapter_restore_uses_original_base_without_double_applying_weights(kind):
    original = tiny_ouro()
    first, restored = copy.deepcopy(original), copy.deepcopy(original)
    cls = ContinualLoRA if kind == "lora" else ContinualOLoRA
    left, right = cls(first, rank=2, dropout=0), cls(restored, rank=2, dropout=0)
    with torch.no_grad():
        for module in left.modules.values():
            module.B.normal_(std=.01)
    left.consolidate()
    right.load_adapter_state(left.adapter_state())
    batch = dict(input_ids=torch.tensor([[1, 3, 4, 2]]), use_cache=False, exit_at_step=3)
    torch.testing.assert_close(first(**batch).logits, restored(**batch).logits, rtol=1e-5, atol=1e-6)
    for controller in (left, right):
        torch.manual_seed(77)
        controller.begin_task(2)
    torch.testing.assert_close(first(**batch).logits, restored(**batch).logits, rtol=1e-5, atol=1e-6)


def test_sequential_lora_uses_formal_setting_and_does_not_change_legacy_config():
    for method in ("full", "osft", "olora", "ewc"):
        config = training_config(parse_args(["train", "--method", method, "--output-dir", "runs/unused"]))
        assert not {"lora_rank", "lora_alpha", "lora_dropout"} & config.keys()
    args = parse_args(["train", "--method", "lora", "--output-dir", "runs/unused", "--resume"])
    assert args.checkpoint_scope == "all" and args.learning_rate == 1e-5
    assert args.micro_batch_size == 8 and args.gradient_accumulation == 16
    assert training_config(args)["lora_rank"] == 8
