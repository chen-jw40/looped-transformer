from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_ouro(model_path: str | Path, loop_count: int, device: str, dtype: torch.dtype | None = None):
    """Set loop depth before construction and assert the runtime copy agrees."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.total_ut_steps = loop_count
    config.use_cache = False
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.config.total_ut_steps = loop_count
    model.model.total_ut_steps = loop_count
    assert model.model.total_ut_steps == loop_count
    return model


def set_loop_count(model, loop_count: int):
    """Set Ouro's runtime loop count on the model and its shared layers."""
    if hasattr(model, "config"):
        model.config.total_ut_steps = loop_count
    changed = 0
    for module in model.modules():
        if hasattr(module, "total_ut_steps"):
            module.total_ut_steps = loop_count
            changed += 1
        if hasattr(module, "config") and hasattr(module.config, "total_ut_steps"):
            module.config.total_ut_steps = loop_count
    if changed == 0:
        raise RuntimeError("Could not find Ouro's runtime total_ut_steps field")
    return changed


def final_exit_loss(model, batch: dict[str, torch.Tensor], loop_count: int):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=None,
        exit_at_step=loop_count - 1,
        use_weighted_exit=False,
        use_cache=False,
        logits_to_keep=0,
    )
    logits = outputs.logits
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    return loss, logits
