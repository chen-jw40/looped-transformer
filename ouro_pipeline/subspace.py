from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ouro_pipeline.projected_full import iter_full_linears
from ouro_pipeline.projection_utils import depth_basis_key


ACTIVATION_SAMPLING_PROTOCOL = {
    "version": "content_stratified_midpoints_v1",
    "eligible_tokens": "attention_mask=1, excluding tokenizer special token IDs",
    "positions": "midpoint of each equal-size stratum of eligible positions; no RNG",
    "alignment": "same token positions across physical modules and loop depths",
}


@dataclass
class BasisUpdate:
    layer: str
    old_rank: int
    added_rank: int
    new_rank: int
    captured_residual_fraction: float
    relative_residual_after_update: float
    saturated: bool
    sampled_vectors: int


class FullDepthActivationCollector:
    """Sample inputs to trainable linear weights at each physical recurrent call.

    The final norm output supplies the inputs to the language-model head and
    exit gate at every depth, including depths where the head is not called.
    """

    def __init__(self, model, max_vectors_per_layer=1024, vectors_per_call=8, scope="all-linear", target_depth=None):
        if max_vectors_per_layer <= 0 or vectors_per_call <= 0:
            raise ValueError("Activation sampling limits must be positive")
        self.max_vectors = max_vectors_per_layer
        self.vectors_per_call = vectors_per_call
        self.target_depth = target_depth
        self.chunks = {}
        self.counts = {}
        self.current_depth = {}
        self.handles = []
        self.valid_positions = None
        self.position_count = None
        selected = dict(iter_full_linears(model, scope=scope))
        core = model.model
        for index, decoder in enumerate(core.layers):
            prefix = f"layers.{index}."

            def mark(_module, _args, kwargs, layer_index=index):
                self.current_depth[layer_index] = int(kwargs["current_ut"])

            self.handles.append(decoder.register_forward_pre_hook(mark, with_kwargs=True))
            for relative_name, module in decoder.named_modules():
                key = prefix + relative_name
                if key not in selected or not isinstance(module, nn.Linear):
                    continue

                def record(_module, args, layer_key=key, layer_index=index):
                    depth = self.current_depth[layer_index]
                    if self.target_depth is None or depth == self.target_depth:
                        self._record(depth_basis_key(layer_key, depth), args[0])

                self.handles.append(module.register_forward_pre_hook(record))
        if scope == "all-linear":
            head_keys = [key for key in ("lm_head", "early_exit_gate") if key in selected]
            if head_keys:
                last_layer = len(core.layers) - 1

                def record_head_inputs(_module, _args, output):
                    depth = self.current_depth[last_layer]
                    if self.target_depth is None or depth == self.target_depth:
                        for key in head_keys:
                            self._record(depth_basis_key(key, depth), output)

                self.handles.append(core.norm.register_forward_hook(record_head_inputs))
        if not selected or len(self.handles) <= len(core.layers):
            self.close()
            raise RuntimeError(f"No trainable linear weights found for scope={scope}")

    def set_batch(self, input_ids, attention_mask, special_token_ids):
        """Restrict this forward pass to non-padding, non-special content tokens."""
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Sampling requires matching [batch, sequence] IDs and attention mask")
        valid = attention_mask.detach().bool().clone()
        for token_id in set(special_token_ids):
            if token_id is not None:
                valid &= input_ids != token_id
        self.position_count = input_ids.numel()
        self.valid_positions = valid.reshape(-1).nonzero().flatten()

    def _record(self, key, value):
        remaining = self.max_vectors - self.counts.get(key, 0)
        if remaining <= 0:
            return
        rows = value.detach().reshape(-1, value.shape[-1])
        if self.valid_positions is None:
            # Tensor-only callers may have no token IDs (e.g. model unit tests).
            valid = torch.arange(rows.shape[0], device=rows.device)
        else:
            if rows.shape[0] != self.position_count:
                raise ValueError(f"{key}: activation positions do not match the sampling batch")
            valid = self.valid_positions
        take = min(remaining, self.vectors_per_call, valid.numel())
        if not take:
            return
        # Integer stratum midpoints avoid the endpoint-only failure at take=2.
        positions = ((2 * torch.arange(take, device=rows.device) + 1) * valid.numel()) // (2 * take)
        indices = valid.index_select(0, positions)
        self.chunks.setdefault(key, []).append(rows.index_select(0, indices).float().cpu())
        self.counts[key] = self.counts.get(key, 0) + take

    def tensors(self):
        return {key: torch.cat(parts) for key, parts in self.chunks.items() if parts}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    q, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return q


def update_basis(
    samples: torch.Tensor,
    old_basis: torch.Tensor | None,
    max_rank: int,
    residual_tolerance: float,
    oversample: int = 8,
) -> tuple[torch.Tensor, dict]:
    """Append dominant residual right-singular directions without deleting old ones."""
    if samples.ndim != 2 or samples.numel() == 0:
        raise ValueError("samples must be a non-empty [vectors, input_dim] matrix")
    if not 0.0 <= residual_tolerance <= 1.0:
        raise ValueError("residual_tolerance must be in [0, 1]")
    dimension = samples.shape[1]
    old = torch.empty(dimension, 0, dtype=torch.float32) if old_basis is None else old_basis.float().cpu()
    old = _orthonormalize(old)
    capacity = max(0, min(max_rank, dimension) - old.shape[1])
    x = samples.float().cpu()
    original_energy = float(x.square().sum())
    residual = x - (x @ old) @ old.transpose(0, 1) if old.numel() else x
    residual_energy = float(residual.square().sum())
    if capacity == 0 or residual_energy <= 1e-20:
        relative = residual_energy / max(original_energy, 1e-20)
        return old, {
            "old_rank": old.shape[1], "added_rank": 0, "new_rank": old.shape[1],
            "captured_residual_fraction": 0.0, "relative_residual_after_update": relative,
            "saturated": capacity == 0 and relative > residual_tolerance,
        }
    numerical_rank = min(residual.shape)
    q = min(numerical_rank, capacity + max(0, oversample))
    _, singular_values, right_vectors = torch.pca_lowrank(residual, q=q, center=False, niter=2)
    order = torch.argsort(singular_values, descending=True)
    singular_values = singular_values[order]
    right_vectors = right_vectors[:, order]
    cumulative = torch.cumsum(singular_values.square(), dim=0)
    target_capture = max(0.0, residual_energy - residual_tolerance * original_energy)
    sufficient = torch.nonzero(cumulative >= target_capture, as_tuple=False)
    if target_capture <= 0:
        add = 0
    elif sufficient.numel():
        add = min(capacity, int(sufficient[0, 0]) + 1)
    else:
        add = capacity
    captured = float(cumulative[add - 1]) if add else 0.0
    new = _orthonormalize(torch.cat([old, right_vectors[:, :add]], dim=1))
    residual_after = max(0.0, residual_energy - captured)
    relative_after = residual_after / max(original_energy, 1e-20)
    return new, {
        "old_rank": old.shape[1], "added_rank": add, "new_rank": new.shape[1],
        "captured_residual_fraction": captured / max(residual_energy, 1e-20),
        "relative_residual_after_update": relative_after,
        "saturated": relative_after > residual_tolerance + 1e-6 and new.shape[1] >= max_rank,
    }


class BasisStore:
    def __init__(self, bases: dict[str, torch.Tensor] | None = None):
        self.bases = {} if bases is None else {key: value.float().cpu() for key, value in bases.items()}

    def update(self, activations, max_rank: int, residual_tolerance: float):
        reports = []
        def order(item):
            layer = int(item.split(".")[1]) if item.startswith("layers.") else 10**9
            depth = int(item.rsplit("@ut", 1)[1]) if "@ut" in item else -1
            return layer, depth, item
        for key in sorted(activations, key=order):
            samples = activations[key]
            if samples.numel() == 0:
                continue
            basis, details = update_basis(samples, self.bases.get(key), max_rank, residual_tolerance)
            self.bases[key] = basis
            reports.append(asdict(BasisUpdate(layer=key, sampled_vectors=samples.shape[0], **details)))
        return reports

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "bases": self.bases}, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported basis checkpoint format")
        return cls(payload["bases"])

    def storage_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self.bases.values())
