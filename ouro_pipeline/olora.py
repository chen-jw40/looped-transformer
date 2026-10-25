"""O-LoRA reference-code variant for shared physical Ouro attention layers.

The released code uses input-side PEFT A factors and an elementwise L1
overlap penalty, rather than the output-side squared penalty in the paper.
No PEFT dependency or historical data replay is needed by this implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


REFERENCE_COMMIT = "07117e1fc4a5f5ad9308a815a42cee8f46502dc8"
REFERENCE_URL = f"https://github.com/cmnfriend/O-LoRA/tree/{REFERENCE_COMMIT}"
PAPER_URL = "https://aclanthology.org/2023.findings-emnlp.715/"
TARGET_SUFFIXES = ("self_attn.q_proj", "self_attn.v_proj")


class OLoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=32.0, dropout=0.1):
        super().__init__()
        if linear.weight.dtype != torch.float32:
            raise ValueError("O-LoRA requires FP32 master weights and factors")
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid O-LoRA rank, alpha or dropout")
        self.base = linear.requires_grad_(False)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rank, self.scaling = rank, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("old_A", linear.weight.new_empty((0, self.in_features)))
        self.register_buffer("old_B", linear.weight.new_empty((self.out_features, 0)))
        self.A = nn.Parameter(linear.weight.new_empty((rank, self.in_features)))
        self.B = nn.Parameter(linear.weight.new_empty((self.out_features, rank)))
        self.reset_current()
        self.train(linear.training)

    @torch.no_grad()
    def reset_current(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)

    def forward(self, x):
        result = self.base(x)
        # Reference code shares one dropout mask between historical/current LoRA.
        adapted = self.dropout(x.to(self.A.dtype))
        if self.old_A.shape[0]:
            result = result + F.linear(F.linear(adapted, self.old_A), self.old_B) * self.scaling
        # FP32 master weights with BF16 autocast require q/k/v outputs to share
        # the native dense projection's dtype (the input may still be FP32).
        return (result + F.linear(F.linear(adapted, self.A), self.B) * self.scaling).to(result.dtype)

    def penalties(self):
        # Keep small inner products/norms in FP32 even during BF16 model autocast.
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            overlap = (self.old_A @ self.A.T).abs().sum()
            norm = torch.linalg.vector_norm(self.A) + torch.linalg.vector_norm(self.B)
        return overlap, norm

    @torch.no_grad()
    def consolidate(self):
        self.old_A = torch.cat((self.old_A, self.A.detach()), dim=0)
        self.old_B = torch.cat((self.old_B, self.B.detach()), dim=1)
        # Remove the current contribution without changing its selected A basis.
        self.B.zero_()
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

    @torch.no_grad()
    def to_linear(self):
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            weight = self.base.weight + self.scaling * (self.old_B @ self.old_A + self.B @ self.A)
        # Meta construction avoids initialization and does not consume training RNG.
        dense = nn.Linear(self.in_features, self.out_features,
                          bias=self.base.bias is not None, device="meta", dtype=weight.dtype)
        dense.weight = nn.Parameter(weight, requires_grad=False)
        if self.base.bias is not None:
            dense.bias = nn.Parameter(self.base.bias.detach().clone(), requires_grad=False)
        dense.train(self.training)
        return dense


class ContinualOLoRA:
    def __init__(self, model, rank=8, alpha=32.0, dropout=0.1,
                 orthogonality_weight=0.5, norm_weight=0.0):
        if any(not math.isfinite(v) or v < 0 for v in (orthogonality_weight, norm_weight)):
            raise ValueError("O-LoRA loss coefficients must be finite and nonnegative")
        self.model = model
        self.orthogonality_weight, self.norm_weight = orthogonality_weight, norm_weight
        targets = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, nn.Linear)
                   and any(name.endswith("." + suffix) for suffix in TARGET_SUFFIXES)]
        if not targets:
            raise ValueError("No physical Ouro q/v projections found")
        model.requires_grad_(False)
        self.modules = {}
        for name, linear in targets:
            parent, _, leaf = name.rpartition(".")
            wrapper = OLoRALinear(linear, rank, alpha, dropout)
            setattr(model.get_submodule(parent), leaf, wrapper)
            self.modules[name] = wrapper
        self.stage = 1
        self.consolidated = False

    def begin_task(self, stage):
        if stage == 1 and self.stage == 1 and not self.consolidated:
            return
        if not self.consolidated or stage != self.stage + 1:
            raise ValueError("O-LoRA tasks must be consolidated and visited in order")
        for module in self.modules.values():
            module.reset_current()
        self.stage, self.consolidated = stage, False

    def penalty_terms(self):
        values = [module.penalties() for module in self.modules.values()]
        return sum(v[0] for v in values), sum(v[1] for v in values)

    def regularization_loss(self):
        overlap, norm = self.penalty_terms()
        return self.orthogonality_weight * overlap + self.norm_weight * norm

    @torch.no_grad()
    def diagnostics(self):
        overlap, norm = self.penalty_terms()
        if not torch.isfinite(overlap) or not torch.isfinite(norm):
            raise FloatingPointError("Non-finite O-LoRA regularization")
        return {"variant": "released-code input-A overlap L1",
                "physical_target_modules": len(self.modules),
                "current_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "historical_rank_per_module": {name: module.old_A.shape[0] for name, module in self.modules.items()},
                "orthogonality_overlap_l1": float(overlap), "current_factor_norm_sum": float(norm),
                "orthogonality_weight": self.orthogonality_weight, "norm_weight": self.norm_weight}

    @torch.no_grad()
    def consolidate(self):
        if self.consolidated:
            raise ValueError("O-LoRA stage already consolidated")
        for module in self.modules.values():
            module.consolidate()
        self.consolidated = True

    def adapter_state(self):
        if not self.consolidated:
            raise ValueError("Restore and consolidate best weights before saving adapters")
        return {"stage": self.stage, "reference_commit": REFERENCE_COMMIT,
                "modules": {name: {"A": module.old_A.detach().cpu().clone(),
                                    "B": module.old_B.detach().cpu().clone(),
                                    "scaling": module.scaling}
                            for name, module in self.modules.items()}}

    @torch.no_grad()
    def load_adapter_state(self, state):
        if state.get("reference_commit") != REFERENCE_COMMIT or state["modules"].keys() != self.modules.keys():
            raise ValueError("Incompatible O-LoRA state")
        for name, module in self.modules.items():
            saved = state["modules"][name]
            if (saved["scaling"] != module.scaling or saved["A"].shape != (state["stage"] * module.rank, module.in_features)
                    or saved["B"].shape != (module.out_features, state["stage"] * module.rank)):
                raise ValueError("O-LoRA rank or scaling changed")
            module.old_A = saved["A"].to(module.A).clone()
            module.old_B = saved["B"].to(module.B).clone()
            module.B.zero_()
            module.A.requires_grad_(False)
            module.B.requires_grad_(False)
        self.stage, self.consolidated = state["stage"], True

    @contextmanager
    def dense_export(self):
        try:
            for name, module in self.modules.items():
                parent, _, leaf = name.rpartition(".")
                setattr(self.model.get_submodule(parent), leaf, module.to_linear())
            yield
        finally:
            for name, module in self.modules.items():
                parent, _, leaf = name.rpartition(".")
                setattr(self.model.get_submodule(parent), leaf, module)
