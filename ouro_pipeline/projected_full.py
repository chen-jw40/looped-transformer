"""Depth-aware historical activation constraints for full weight matrices."""
from __future__ import annotations

import re

import torch
from torch import nn

from ouro_pipeline.projection_utils import union_basis


_DECODER_LINEAR = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def canonical_full_linear_name(name):
    match = _DECODER_LINEAR.search(name)
    if match is not None:
        return f"layers.{int(match.group(1))}.{match.group(2)}"
    if name == "lm_head" or name.endswith(".lm_head"):
        return "lm_head"
    if name == "early_exit_gate" or name.endswith(".early_exit_gate"):
        return "early_exit_gate"
    raise ValueError(f"Not a supported Ouro linear: {name}")


def iter_full_linears(model, scope="all-linear"):
    if scope not in ("all-linear", "o-proj"):
        raise ValueError(f"Unknown projection scope: {scope}")
    seen = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not module.weight.requires_grad:
            continue
        try:
            key = canonical_full_linear_name(name)
        except ValueError:
            continue
        if scope == "o-proj" and not key.endswith("self_attn.o_proj"):
            continue
        if key in seen:
            raise ValueError(f"Duplicate physical linear module: {key}")
        seen.add(key)
        yield key, module


class HistoricalFullWeightProjection:
    """After AdamW, enforce W U = W_task_entry U for each selected linear.

    U is the orthogonal union of historical depth banks, or a preselected joint
    basis. This local invariant does not constrain
    non-linear parameters or guarantee unchanged recurrent-network outputs.
    """

    def __init__(self, model, bases, scope="all-linear", joint_bases=None):
        if joint_bases is not None and bases:
            raise ValueError("Provide either depth banks or joint bases, not both")
        self.constraints = []
        self.joint = joint_bases is not None
        self.steps = 0
        self.max_relative_error = 0.0
        self.max_absolute_error = 0.0
        self.source_depths = sorted({int(key.rsplit("@ut", 1)[1]) for key, basis in bases.items()
                                     if "@ut" in key and basis.numel()})
        for key, module in iter_full_linears(model, scope=scope):
            basis = joint_bases.get(key) if self.joint else union_basis(bases, key)
            if basis is None:
                continue
            if self.joint:
                if not torch.isfinite(basis).all() or basis.ndim != 2:
                    raise ValueError(f"{key}: invalid joint basis")
                identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
                if not torch.allclose(basis.T @ basis, identity, atol=1e-4, rtol=1e-4):
                    raise ValueError(f"{key}: joint basis must be orthonormal")
                if not basis.numel():
                    continue
            weight = module.weight
            if weight.dtype != torch.float32:
                raise ValueError(f"{key}: strict full-weight projection requires FP32 weights")
            u = basis.to(device=weight.device, dtype=weight.dtype)
            if weight.shape[1] != u.shape[0]:
                raise ValueError(f"{key}: basis input dimension {u.shape[0]} != {weight.shape[1]}")
            target = weight.detach() @ u
            bias = module.bias
            bias_target = None if bias is None else bias.detach().clone()
            self.constraints.append((key, weight, u, target, bias, bias_target))

    @torch.no_grad()
    def after_step(self):
        for key, weight, u, target, bias, bias_target in self.constraints:
            # A second pass reduces FP32 accumulation residuals.
            for _ in range(2):
                residual = weight @ u - target
                weight.addmm_(residual, u.T, beta=1.0, alpha=-1.0)
            if bias is not None:
                bias.copy_(bias_target)
            residual = weight @ u - target
            error = torch.linalg.vector_norm(residual)
            scale = torch.linalg.vector_norm(target).clamp_min(1.0)
            relative = float(error / scale)
            self.max_relative_error = max(self.max_relative_error, relative)
            self.max_absolute_error = max(self.max_absolute_error, float(error))
            if not torch.isfinite(error):
                raise FloatingPointError(f"{key}: non-finite full-weight history constraint error")
        self.steps += 1

    def diagnostics(self):
        return {
            "optimizer_steps_checked": self.steps,
            "protected_linear_weights": len(self.constraints),
            "historical_basis_depths": self.source_depths,
            "basis_representation": "joint" if self.joint else "depth_banks",
            "history_union_ranks": {key: u.shape[1] for key, _, u, _, _, _ in self.constraints},
            "max_WU_history_scaled_error": self.max_relative_error,
            "max_WU_history_absolute_error": self.max_absolute_error,
        }
