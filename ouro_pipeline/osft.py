"""Weight-SVD OSFT for shared physical Ouro layers, independent of loop depth.

Implements the task-budget variant in Sculpting Subspaces, Appendix A.10.
The gradient/parameter projections follow the mathematics of mini_trainer
commit fd5b552177bde9a551d30e73660d6a96205c97c0 (optim_wrapper).
This is a local implementation; no mini_trainer dependency is required.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F


PAPER_URL = "https://proceedings.iclr.cc/paper_files/paper/2026/file/22620cccae29bbb3f18d226ea0320ff5-Paper-Conference.pdf"
REFERENCE_COMMIT = "fd5b552177bde9a551d30e73660d6a96205c97c0"
REFERENCE_URL = f"https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/blob/{REFERENCE_COMMIT}/src/mini_trainer/osft_utils.py"
TARGET_SUFFIXES = tuple(f"{group}.{name}" for group, names in (
    ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
) for name in names)


def retention_ratio(stage, total_tasks):
    if not 1 <= stage <= total_tasks:
        raise ValueError("OSFT requires 1 <= stage <= total_tasks")
    return (stage - 1) / total_tasks


def target_linears(model):
    """One target per physical module; embedding, head, gate and norms excluded."""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and any(name.endswith("." + suffix) for suffix in TARGET_SUFFIXES)]


class OSFTLinear(nn.Module):
    """Frozen leading SVD factors plus trainable trailing factors (V stored as rows)."""

    def __init__(self, linear, protected_rank):
        super().__init__()
        if linear.weight.dtype != torch.float32:
            raise ValueError("OSFT master weights and factors must be FP32")
        rank = min(linear.weight.shape)
        if not 0 < protected_rank < rank:
            raise ValueError("OSFTLinear requires 0 < protected_rank < matrix rank")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.protected_rank = protected_rank
        # NGC containers may enable TF32 by default. Subspace reconstruction
        # and projections need IEEE FP32; BF16 autocast linear products remain.
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad(), torch.autocast(device_type=linear.weight.device.type, enabled=False):
            u, s, vt = torch.linalg.svd(linear.weight.detach(), full_matrices=False,
                                      **({"driver": "gesvd"} if linear.weight.is_cuda else {}))
        # Clone slices so each factor owns its storage and discarded SVD tensors free.
        self.register_buffer("U_high", u[:, :protected_rank].clone())
        self.register_buffer("S_high", s[:protected_rank].clone())
        self.register_buffer("V_high", vt[:protected_rank].clone())
        self.U_low = nn.Parameter(u[:, protected_rank:].clone())
        self.S_low = nn.Parameter(s[protected_rank:].clone())
        self.V_low = nn.Parameter(vt[protected_rank:].clone())
        self.register_parameter("bias", linear.bias)
        with torch.no_grad():
            error = torch.linalg.vector_norm(self.dense_weight() - linear.weight)
            scale = torch.linalg.vector_norm(linear.weight).clamp_min(1e-12)
            self.reconstruction_relative_error = float(error / scale)
        if not math.isfinite(self.reconstruction_relative_error) or self.reconstruction_relative_error > 1e-4:
            raise FloatingPointError(f"OSFT SVD reconstruction error: {self.reconstruction_relative_error}")
        self.train(linear.training)

    def dense_weight(self):
        # Reconstruct in FP32, then let F.linear use the caller's BF16 autocast.
        # This preserves the existing dense-linear forward/loss convention.
        with torch.autocast(device_type=self.U_low.device.type, enabled=False):
            return ((self.U_high * self.S_high) @ self.V_high
                    + (self.U_low * self.S_low) @ self.V_low)

    def forward(self, x):
        return F.linear(x, self.dense_weight(), self.bias)

    @torch.no_grad()
    def project_gradients(self):
        if self.U_low.grad is not None:
            self.U_low.grad.sub_(self.U_high @ (self.U_high.T @ self.U_low.grad))
        if self.V_low.grad is not None:
            self.V_low.grad.sub_((self.V_low.grad @ self.V_high.T) @ self.V_high)

    @torch.no_grad()
    def project_parameters(self):
        # AdamW's elementwise preconditioning can leave the allowed subspace.
        self.U_low.sub_(self.U_high @ (self.U_high.T @ self.U_low))
        self.V_low.sub_((self.V_low @ self.V_high.T) @ self.V_high)

    @torch.no_grad()
    def orthogonality_errors(self):
        u_error = torch.linalg.vector_norm(self.U_high.T @ self.U_low)
        v_error = torch.linalg.vector_norm(self.V_low @ self.V_high.T)
        return (float(u_error / torch.linalg.vector_norm(self.U_low).clamp_min(1e-12)),
                float(v_error / torch.linalg.vector_norm(self.V_low).clamp_min(1e-12)))

    @torch.no_grad()
    def to_linear(self):
        # Meta construction performs no real random initialization/allocation.
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device="meta", dtype=torch.float32)
        linear.weight = nn.Parameter(self.dense_weight().detach())
        linear.bias = self.bias
        return linear.train(self.training)


class StageOSFT:
    """Install for one task, project around AdamW steps, then export dense weights."""

    def __init__(self, model, stage, total_tasks, tolerance=1e-4):
        self.model, self.stage, self.total_tasks = model, stage, total_tasks
        self.ratio = retention_ratio(stage, total_tasks)
        self.tolerance = tolerance
        self.modules = {}
        self.ranks = {}
        self.steps = 0
        self.materialized = False
        started = time.perf_counter()
        # Keep names only, so old dense modules are released incrementally.
        names = [name for name, _ in target_linears(model)]
        if not names:
            raise ValueError("OSFT found no attention/MLP linear weights")
        target_ids = {id(model.get_submodule(name).weight) for name in names}
        self.non_osft_parameter_names = [name for name, parameter in model.named_parameters()
                                        if parameter.requires_grad and id(parameter) not in target_ids]
        self.dense_model_parameters = sum(p.numel() for p in model.parameters())
        for index, name in enumerate(names, 1):
            linear = model.get_submodule(name)
            rank = min(linear.weight.shape)
            k = math.floor(self.ratio * rank)
            self.ranks[name] = {"shape": list(linear.weight.shape), "matrix_rank": rank,
                                "protected_rank": k, "trainable_rank": rank - k}
            if k:
                try:
                    wrapper = OSFTLinear(linear, k)
                except Exception as error:
                    raise RuntimeError(f"OSFT SVD initialization failed for {name}: {error}") from error
                parent_name, child = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), child, wrapper)
                self.modules[name] = wrapper
                if index == 1 or index % 24 == 0 or index == len(names):
                    print(f"OSFT stage {stage}: SVD {index}/{len(names)}, {name}, protected_rank={k}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.svd_seconds = time.perf_counter() - started
        self.trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.frozen_factor_elements = sum(b.numel() for m in self.modules.values() for b in m.buffers())

    def before_step(self):
        for module in self.modules.values():
            module.project_gradients()

    def after_step(self):
        for module in self.modules.values():
            module.project_parameters()
        self.steps += 1

    def diagnostics(self):
        errors = {name: module.orthogonality_errors() for name, module in self.modules.items()}
        if any(not math.isfinite(value) for values in errors.values() for value in values):
            raise FloatingPointError("OSFT non-finite orthogonality residual")
        if any(not torch.isfinite(parameter).all()
               for module in self.modules.values() for parameter in module.parameters()):
            raise FloatingPointError("OSFT non-finite factor or bias")
        worst = max((max(value) for value in errors.values()), default=0.0)
        if not math.isfinite(worst) or worst > self.tolerance:
            raise FloatingPointError(f"OSFT final relative orthogonality residual {worst} > {self.tolerance}")
        return {"stage": self.stage, "total_tasks": self.total_tasks,
                "retention_ratio": self.ratio, "rank_schedule": "(stage - 1) / total_tasks",
                "target_matrix_count": len(self.ranks), "decomposed_matrix_count": len(self.modules),
                "ranks": self.ranks, "optimizer_steps_projected": self.steps,
                "svd_seconds": self.svd_seconds,
                "max_initial_reconstruction_relative_error": max(
                    (m.reconstruction_relative_error for m in self.modules.values()), default=0.0),
                "final_relative_orthogonality_errors": {name: {"U": u, "V": v} for name, (u, v) in errors.items()},
                "max_final_relative_orthogonality_error": worst,
                "orthogonality_tolerance": self.tolerance,
                "dense_model_parameters": self.dense_model_parameters,
                "trainable_parameters": self.trainable_parameters,
                "frozen_factor_elements": self.frozen_factor_elements,
                "non_osft_parameter_names": self.non_osft_parameter_names,
                "loop_depth_specific": False,
                "post_adamw_parameter_projection": True}

    def materialize(self):
        if self.materialized:
            raise RuntimeError("OSFT stage already materialized")
        for name, module in self.modules.items():
            parent_name, child = name.rsplit(".", 1)
            setattr(self.model.get_submodule(parent_name), child, module.to_linear())
        self.modules.clear()
        self.materialized = True
