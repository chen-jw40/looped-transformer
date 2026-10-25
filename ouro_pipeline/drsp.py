"""DRSP from sampled, ordinary SFT backwards; one spectral history per W.

R is shared by all recurrent calls of a physical module within a microbatch.
The bounded current-task buffer contains at most samples * (D+1) * width rows;
only rank-k eigenvectors AND their eigenvalues persist across tasks. We merge
per-task mean second moments without trace normalization or unit-basis pooling.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import torch

from ouro_pipeline.adaptive_subspace import _svd
from ouro_pipeline.projected_full import iter_full_linears


def drsp_rows(sketches, strength):
    """Z: [D,n,s] -> rows whose Gram is barZ barZ.T + lambda/D sum H H.T."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("DRSP lambda must be finite and nonnegative")
    mean = sketches.mean(0)
    pieces = [mean.T]
    if strength:
        pieces.extend((z - mean).T * math.sqrt(strength / len(sketches)) for z in sketches)
    return torch.cat(pieces)


class DRSPCollector:
    def __init__(self, model, loops, samples=64, width=8, strength=.5, seed=42, scope="all-linear"):
        if min(loops, samples, width) <= 0 or not math.isfinite(strength) or strength < 0:
            raise ValueError("Invalid DRSP sketch configuration")
        self.loops, self.samples, self.width, self.strength = loops, samples, width, strength
        self.modules = dict(iter_full_linears(model, scope))
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.handles, self.tensor_handles = [], []
        self.recording = False
        self.depths, self.chunks, self.energies = {}, {}, {}
        self.indices, self.count, self.seconds, self.answer_tokens = [], 0, 0., 0
        self.example_indices = []
        for index, decoder in enumerate(model.model.layers):
            def mark(_module, _args, kwargs, index=index):
                if self.recording:
                    self.depths[index] = int(kwargs["current_ut"])
            self.handles.append(decoder.register_forward_pre_hook(mark, with_kwargs=True))
        for key, module in self.modules.items():
            def record(_module, inputs, output, key=key):
                if self.recording:
                    depth = self.depths[int(key.split(".")[1])] if key.startswith("layers.") else len(self.calls.get(key, {}))
                    self._record(key, depth, inputs[0], output)
            self.handles.append(module.register_forward_hook(record))

    def configure(self, total_micro_steps):
        take = min(self.samples, total_micro_steps)
        # Deterministic coverage of the planned trajectory, including its ends.
        self.schedule = {i * (total_micro_steps - 1) // max(take - 1, 1) for i in range(take)}

    def begin(self, micro_step, batch, accumulation):
        if micro_step not in self.schedule:
            return
        if self.tensor_handles or self.recording:
            raise RuntimeError("DRSP sample not finalized")
        self.started = time.perf_counter()
        self.calls, self.probes, self.pending, self.backward_seen = {}, {}, {}, {}
        # One example per sampled microbatch keeps the original bounded sample
        # budget. Rotate slots in the shuffled batch without using training RNG.
        self.example_index = self.count % batch["input_ids"].shape[0]
        tokens = (batch["labels"][:, 1:] != -100).sum(dim=1)
        selected_tokens = int(tokens[self.example_index])
        if selected_tokens <= 0:
            raise ValueError("DRSP sampled an example without answer tokens")
        # Batch CE averages over all answer tokens. Recover this example's own
        # token-mean CE adjoint, also undoing gradient accumulation.
        self.scale = accumulation * int(tokens.sum()) / selected_tokens
        self.indices.append(micro_step)
        self.example_indices.append(self.example_index)
        self.answer_tokens += selected_tokens
        self.recording = True

    def _record(self, key, depth, inputs, output):
        calls = self.calls.setdefault(key, {})
        if depth in calls:
            raise RuntimeError(f"{key}: duplicate DRSP forward depth {depth}")
        calls[depth] = True
        if key not in self.probes:
            self.probes[key] = torch.randint(0, 2, (output.shape[-1], self.width),
                device=output.device, generator=self.generator, dtype=torch.int8).float().mul_(2).sub_(1).div_(math.sqrt(self.width))
        # Retain only one example, on host memory. Holding complete batch inputs
        # would defeat activation checkpointing at batch8/4096 tokens.
        x = inputs.detach()[self.example_index].reshape(-1, inputs.shape[-1]).to("cpu", copy=True)
        if not output.requires_grad:
            raise RuntimeError("DRSP requires a graph and non-reentrant checkpointing")

        def adjoint(grad):
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=grad.device.type, enabled=False):
                seen = self.backward_seen.setdefault(key, set())
                if depth in seen:
                    raise RuntimeError(f"{key}: duplicate DRSP backward depth {depth}")
                seen.add(depth)
                # Undo loss/accumulation, and sum ALL token contributions before
                # forming a covariance; no per-token covariance substitution.
                b = grad.detach()[self.example_index].reshape(-1, grad.shape[-1]).float()
                z = x.to(device=grad.device, dtype=torch.float32).T @ (b @ self.probes[key])
                z.mul_(self.scale)
                self.pending.setdefault(key, {})[depth] = z
            self.seconds += time.perf_counter() - started
            return None
        self.tensor_handles.append(output.register_hook(adjoint))

    def before_backward(self):
        # Checkpoint recomputation must not register additional call hooks.
        self.recording = False

    @torch.no_grad()
    def finish(self):
        if not self.tensor_handles:
            return
        started = time.perf_counter()
        for key, module in self.modules.items():
            calls = self.calls.get(key, {})
            if not calls:
                raise RuntimeError(f"{key}: missing DRSP forward")
            seen = self.backward_seen.get(key, set())
            if key.startswith("layers.") and (set(calls) != set(range(self.loops)) or seen != set(calls)):
                raise RuntimeError(f"{key}: incomplete recurrent gradients: {seen}")
            # Ouro's unused exit gate has no final-CE adjoint: true zero signal,
            # not fabricated final-depth activations. lm_head has D=1.
            zero = lambda: torch.zeros(module.in_features, self.width, device=self.device)
            zs = torch.stack([self.pending.get(key, {}).get(d, None) if d in seen else zero() for d in sorted(calls)])
            rows = drsp_rows(zs, self.strength)
            if not torch.isfinite(rows).all():
                raise FloatingPointError(f"{key}: non-finite DRSP sketch")
            energy = self.energies.setdefault(key, {"mean": 0., "variation": 0., "covariance": 0., "calls_without_adjoint": 0})
            energy["mean"] += float(zs.mean(0).square().sum())
            energy["variation"] += float((zs - zs.mean(0)).square().sum() / len(zs))
            energy["covariance"] += float(rows.square().sum())
            energy["calls_without_adjoint"] += len(calls) - len(seen)
            self.chunks.setdefault(key, []).append(rows.cpu())
        self.count += 1
        for handle in self.tensor_handles:
            handle.remove()
        self.tensor_handles.clear()
        self.pending.clear()
        self.probes.clear()
        self.seconds += time.perf_counter() - started

    def tensors(self):
        if not self.count or self.modules.keys() != self.chunks.keys():
            raise RuntimeError("No complete DRSP trajectory statistics")
        return {key: torch.cat(rows) / math.sqrt(self.count) for key, rows in self.chunks.items()}

    def diagnostics(self):
        return {"status": "complete", "samples": self.count, "planned_samples": len(self.schedule),
                "micro_steps": self.indices, "answer_tokens": self.answer_tokens,
                "example_indices": self.example_indices,
                "sampling": "one example per scheduled microbatch; rotating batch slot",
                "sketch_width": self.width, "lambda": self.strength,
                "extra_backward_passes": 0, "statistic": "raw per-example final-CE call gradients, accumulation undone",
                "weights": "sampled training trajectory, including steps after selected best eval-CE checkpoint",
                "probe_sharing": "same R across all calls of each physical module within each sample",
                "energy_by_module": self.energies, "hook_cpu_wall_seconds": self.seconds}

    def state_dict(self):
        if self.recording or self.tensor_handles:
            raise RuntimeError("DRSP recovery requires a finalized backward")
        return {"protocol": (self.loops, self.samples, self.width, self.strength),
                "schedule": sorted(self.schedule), "generator": self.generator.get_state(),
                **{key: getattr(self, key) for key in
                   ("chunks", "energies", "indices", "example_indices", "count", "seconds", "answer_tokens")}}

    def load_state_dict(self, state):
        if (state["protocol"] != (self.loops, self.samples, self.width, self.strength)
                or state["schedule"] != sorted(self.schedule)):
            raise ValueError("DRSP recovery sampling protocol differs")
        self.generator.set_state(state["generator"].cpu())
        for key in ("chunks", "energies", "indices", "example_indices", "count", "seconds", "answer_tokens"):
            setattr(self, key, state[key])

    def close(self):
        self.recording = False
        for handle in self.handles + self.tensor_handles:
            handle.remove()
        self.handles.clear()
        self.tensor_handles.clear()


class DRSPBasisStore:
    def __init__(self, rank, method="randomized"):
        self.rank, self.method, self.history, self.bases = rank, method, {}, {}

    @torch.no_grad()
    def update(self, factors, device="cpu"):
        if not factors or (self.history and self.history.keys() != factors.keys()):
            raise ValueError("DRSP task module keys changed")
        reports = []
        for key, rows in factors.items():
            started = time.perf_counter()
            if rows.ndim != 2 or not torch.isfinite(rows).all():
                raise ValueError(f"{key}: invalid DRSP factor")
            factor = rows.T.to(device=device, dtype=torch.float32)
            task_trace = float(factor.square().sum())
            previous = self.history.get(key)
            trace, count = task_trace, 1
            if previous is not None:
                old = previous["basis"].to(device) * previous["eigenvalues"].to(device).sqrt()
                factor = torch.cat([old, factor], dim=1)
                trace += previous["full_trace"]
                count += previous["task_count"]
            q, s = _svd(factor, self.rank, self.method)
            if q.shape[1]:
                # CUDA's randomized SVD can return measurably nonorthogonal U
                # at large ranks. Preserve U diag(s)^2 U.T while repairing its
                # coordinates; a bare QR would misassign the spectral weights.
                q, r = torch.linalg.qr(q, mode="reduced")
                options = {"driver": "gesvd"} if q.is_cuda else {}
                rotation, s, _ = torch.linalg.svd(r * s, full_matrices=False, **options)
                q = q @ rotation
            self.bases[key] = q.cpu()
            self.history[key] = {"basis": self.bases[key], "eigenvalues": s.square().cpu(),
                                 "full_trace": trace, "task_count": count}
            reports.append({"layer": key, "new_rank": q.shape[1], "task_count": count,
                            "task_trace": task_trace, "historical_full_trace": trace,
                            "retained_trace": float(s.square().sum()), "seconds": time.perf_counter() - started})
        return reports

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": "drsp_v1", "rank": self.rank, "method": self.method,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] != "drsp_v1":
            raise ValueError("Unsupported DRSP history")
        store = cls(state["rank"], state["method"])
        store.history = state["history"]
        store.bases = {key: h["basis"] for key, h in store.history.items()}
        return store

    def storage_bytes(self):
        return sum(h[k].numel() * h[k].element_size() for h in self.history.values() for k in ("basis", "eigenvalues"))
