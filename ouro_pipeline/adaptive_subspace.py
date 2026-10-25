"""Task-balanced historical spectra and a bounded joint activation protection basis.

Histories are compressed uncentered second moments, not permanent constraints.
Only the selected joint basis constrains the next task; discarded directions
may become unprotected. Persistent tensors live on CPU, work is module-wise.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch

BALANCED_SELECTORS = {"joint_balanced"}
SELECTORS = {"joint_greedy", "joint_local", "joint_pca"} | BALANCED_SELECTORS


def _svd(factor, rank, method):
    rank = min(rank, *factor.shape)
    if method == "exact" or rank + 8 >= min(factor.shape):
        u, s, _ = torch.linalg.svd(factor, full_matrices=False)
    elif method == "randomized":
        u, s, _ = torch.svd_lowrank(factor, q=min(rank + 8, *factor.shape), niter=2)
        order = s.argsort(descending=True)
        u, s = u[:, order], s[order]
    else:
        raise ValueError(f"Unknown SVD method: {method}")
    # Avoid inventing directions when the factor is numerically rank deficient.
    keep = min(rank, int((s > s[0] * 1e-6).sum())) if s.numel() else 0
    return u[:, :keep].contiguous(), s[:keep]


@torch.no_grad()
def update_history(previous, x, history_rank, method="randomized"):
    if history_rank <= 0 or x.ndim != 2 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonempty activation rows and a positive history rank")
    x = x.float()
    norm = torch.linalg.vector_norm(x)
    if norm <= 0:
        raise ValueError("Cannot normalize zero-energy activations")
    count = 1 if previous is None else previous["task_count"] + 1
    factor = x.T / norm / count**0.5
    if previous is not None:
        old = previous["basis"].to(x.device)
        eigenvalues = previous["eigenvalues"].to(x.device)
        if old.shape[0] != x.shape[1]:
            raise ValueError("History input dimension changed")
        factor = torch.cat([old * (eigenvalues * ((count - 1) / count)).sqrt(), factor], dim=1)
    u, s = _svd(factor, history_rank, method)
    # Full trace includes energy discarded at previous compression steps.
    trace = 1.0 if previous is None else ((count - 1) * previous["full_trace"] + 1.0) / count
    return {"basis": u.cpu(), "eigenvalues": s.square().cpu(),
            "full_trace": trace, "task_count": count}


@torch.no_grad()
def allocate_joint(histories, rank_budget, selector="joint_greedy", device="cpu", method="randomized", balance_epsilon=.01):
    if not histories or rank_budget <= 0 or selector not in SELECTORS:
        raise ValueError("Expected histories, a positive budget and a joint selector")
    if not 0 < balance_epsilon <= 1:
        raise ValueError("Balance epsilon must be in (0, 1]")
    directions = [h["basis"].to(device=device, dtype=torch.float32) for h in histories]
    a = torch.cat(directions, dim=1)
    weights = torch.cat([h["eigenvalues"].to(device) / (len(histories) * max(h["full_trace"], 1e-30))
                         for h in histories])
    budget = min(rank_budget, a.shape[0], a.shape[1])
    source_counts = None
    if selector == "joint_pca":
        q, _ = _svd(a * weights.sqrt(), budget, method)
    else:
        # G contains candidate residual inner products after GLOBAL projection.
        # joint_greedy sums weight_i G_ij^2 / G_jj over ALL depths;
        # joint_local only sums over i from candidate j's source depth.
        gram = a.T @ a
        q = a.new_empty((a.shape[0], 0))
        available = torch.ones(a.shape[1], dtype=torch.bool, device=device)
        origins = [k for k, u in enumerate(directions) for _ in range(u.shape[1])]
        source_counts = [0] * len(histories)
        if selector == "joint_local":
            depth_slices = []
            start = 0
            for u in directions:
                depth_slices.append(slice(start, start + u.shape[1]))
                start += u.shape[1]
        if selector in BALANCED_SELECTORS:
            depth_energy = a.new_zeros((len(histories), a.shape[1]))
            start = 0
            for d, (u, h) in enumerate(zip(directions, histories)):
                depth_energy[d, start:start + u.shape[1]] = h["eigenvalues"].to(device) / max(h["full_trace"], 1e-30)
                start += u.shape[1]
            alpha = a.new_full((len(histories),), 1 / len(histories))
            covered = a.new_zeros(len(histories))
        for _ in range(a.shape[1]):
            if q.shape[1] >= budget:
                break
            diagonal = gram.diagonal()
            valid = available & (diagonal > 1e-6)
            if not valid.any():
                break
            if selector in BALANCED_SELECTORS:
                gains = (depth_energy @ gram.square()) / diagonal.clamp_min(1e-6)[None, :]
                score = (alpha[:, None] * torch.log1p(gains / (balance_epsilon + covered[:, None]))).sum(0)
            elif selector == "joint_local":
                score = a.new_empty(a.shape[1])
                for block in depth_slices:
                    score[block] = ((weights[block, None] * gram[block, block].square()).sum(0)
                                    / diagonal[block].clamp_min(1e-6))
            else:
                score = (weights[:, None] * gram.square()).sum(0) / diagonal.clamp_min(1e-6)
            score.masked_fill_(~valid, -torch.inf)
            index = int(score.argmax())
            available[index] = False
            v = a[:, index].clone()
            for _ in range(2):
                v -= q @ (q.T @ v)
            norm = torch.linalg.vector_norm(v)
            if norm.square() <= 1e-6:
                continue
            v /= norm
            q = torch.cat([q, v[:, None]], dim=1)
            source_counts[origins[index]] += 1
            b = a.T @ v
            if selector in BALANCED_SELECTORS:
                covered += depth_energy @ b.square()
            gram -= torch.outer(b, b)
    coverage = [float(((u.T @ q).square().sum(1) * h["eigenvalues"].to(device)).sum()
                      / max(h["full_trace"], 1e-30)) for u, h in zip(directions, histories)]
    details = {"source_counts": source_counts, "coverage": coverage,
               "joint_coverage": sum(coverage) / len(coverage), "new_rank": q.shape[1]}
    if selector == "joint_local":
        details["selection_gain_scope"] = "source_depth_only"
    if selector in BALANCED_SELECTORS:
        actual = a.new_tensor(coverage)
        details.update(depth_weights=alpha.tolist(), balance_epsilon=balance_epsilon,
                       balanced_objective=float((alpha * (balance_epsilon + actual).log()).sum()),
                       weighted_joint_coverage=float((alpha * actual).sum()))
    return q.cpu(), details


class AdaptiveBasisStore:
    def __init__(self, history_rank=512, selector="joint_greedy", svd_method="randomized", balance_epsilon=.01):
        if history_rank <= 0 or selector not in SELECTORS:
            raise ValueError("Invalid adaptive history configuration")
        if svd_method not in {"exact", "randomized"}:
            raise ValueError("Invalid SVD method")
        self.history_rank, self.selector, self.svd_method = history_rank, selector, svd_method
        if not 0 < balance_epsilon <= 1:
            raise ValueError("Balance epsilon must be in (0, 1]")
        self.balance_epsilon = balance_epsilon
        self.history = {}
        self.bases = {}

    def update(self, activations, max_rank, device="cpu"):
        if not activations or (self.history and activations.keys() != self.history.keys()):
            raise ValueError("Every task must provide the same module/depth activation keys")
        grouped = {}
        for key in sorted(activations):
            module, depth = key.rsplit("@ut", 1)
            grouped.setdefault(module, []).append((int(depth), key))
        history, bases, reports = {}, {}, []
        for module, entries in grouped.items():
            started = time.perf_counter()
            entries.sort()
            values = {key: activations[key] for _, key in entries}
            for _, key in entries:
                history[key] = update_history(self.history.get(key), values[key].to(device),
                                              self.history_rank, self.svd_method)
            history_seconds = time.perf_counter() - started
            q, details = allocate_joint([history[key] for _, key in entries], max_rank,
                                        self.selector, device, self.svd_method, self.balance_epsilon)
            bases[module] = q
            reports.append({"layer": module, "source_depths": [d for d, _ in entries],
                            "old_rank": self.bases[module].shape[1] if module in self.bases else 0,
                            "sampled_vectors": sum(activations[key].shape[0] for _, key in entries),
                            "history_retained_trace": [float(history[key]["eigenvalues"].sum())
                                                       for _, key in entries],
                            "history_seconds": history_seconds,
                            "allocation_seconds": time.perf_counter() - started - history_seconds,
                            **details})
        self.history, self.bases = history, bases
        return reports

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": "adaptive_v1", "history_rank": self.history_rank,
                    "selector": self.selector, "svd_method": self.svd_method,
                    **({"balance_epsilon": self.balance_epsilon} if self.selector in BALANCED_SELECTORS else {}),
                    "history": self.history, "bases": self.bases}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != "adaptive_v1":
            raise ValueError("Unsupported adaptive history checkpoint")
        store = cls(state["history_rank"], state["selector"], state["svd_method"], state.get("balance_epsilon", .01))
        store.history, store.bases = state["history"], state["bases"]
        return store

    def storage_breakdown(self):
        size = lambda t: t.numel() * t.element_size()
        return {"protection_bytes": sum(size(q) for q in self.bases.values()),
                "history_bytes": sum(size(h["basis"]) + size(h["eigenvalues"])
                                     for h in self.history.values())}

    def storage_bytes(self):
        return sum(self.storage_breakdown().values())
