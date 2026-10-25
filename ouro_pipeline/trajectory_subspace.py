"""Last-anchored, forward-only selection from paired loop trajectories.

Keep every old constraint and the rank of a same-weight last/append reference.
Within that budget, improve normalized adjacent-depth difference coverage while
enforcing the last-depth energy floor. No gradients or loss probes are used.
The eigenspace search is approximate; a feasible last reference is always kept.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

import torch

from ouro_pipeline.adaptive_subspace import _svd
from ouro_pipeline.subspace import update_basis


@contextmanager
def _accurate_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _energy(rows, basis):
    # FP64 reductions avoid cancellation when comparing near-identical coverage.
    return float((rows @ basis).double().square().sum())


@torch.no_grad()
def select_trajectory(xs, old, max_rank, residual_tolerance=.05, *, device="cpu",
                      method="randomized", floor="tolerance", head=False):
    if floor not in {"tolerance", "reference"}:
        raise ValueError("Trajectory floor must be tolerance or reference")
    if len(xs) < 2 or any(x.ndim != 2 or x.shape != xs[-1].shape or not x.numel()
                         or not torch.isfinite(x).all() for x in xs):
        raise ValueError("Expected finite, nonempty, position-paired depth matrices")
    last = xs[-1].float().cpu()
    previous = last.new_empty((last.shape[1], 0)) if old is None else old.float().cpu()
    reference, details = update_basis(last, previous, max_rank, residual_tolerance)
    added = reference.shape[1] - previous.shape[1]
    total_energy = float(last.double().square().sum())
    with _accurate_matmul():
        reference_captured = _energy(last, reference)
        reference_residual = max(0., total_energy - reference_captured)
        # If last itself cannot meet the tolerance within the rank cap, retain
        # its measured coverage floor instead of imposing an infeasible target.
        allowed_residual = (reference_residual if floor == "reference" else
                            max(residual_tolerance * total_energy, reference_residual))
        required_captured = max(0., total_energy - allowed_residual)
        diagnostics = {**details, "reference_rank": reference.shape[1],
            "reference_last_residual_fraction": reference_residual / max(total_energy, 1e-30),
            "last_residual_limit": allowed_residual / max(total_energy, 1e-30),
            "last_residual_fraction": reference_residual / max(total_energy, 1e-30),
            "floor_policy": floor, "trajectory_coverage_before": None,
            "trajectory_coverage_after": None, "solver_evaluations": 0,
            "solver_multiplier": None, "candidate_rank": added,
            "selection": "head_last" if head else "last_fallback", "floor_satisfied": True}
        if head or added <= 0 or total_energy <= 0:
            diagnostics["selection"] = "head_last" if head else "no_new_capacity"
            return reference, diagnostics

        old_device = previous.to(device)
        reference_device = reference.to(device)
        xlast = last.to(device)
        residual_last = xlast - (xlast @ old_device) @ old_device.T
        differences = []
        for left, right in zip(xs[:-1], xs[1:]):
            difference = right.float().to(device) - left.float().to(device)
            norm = torch.linalg.vector_norm(difference)
            # Normalize by full difference energy, not post-projection energy:
            # old constraints already capture part of each trajectory.
            if norm > 0:
                differences.append(difference / norm / (len(xs) - 1)**.5)
        if not differences:
            diagnostics["selection"] = "zero_trajectory"
            return reference, diagnostics
        trajectory = torch.cat(differences)
        residual_trajectory = trajectory - (trajectory @ reference_device) @ reference_device.T
        extra_rank = min(added, last.shape[1] - reference.shape[1], *residual_trajectory.shape)
        extra = residual_trajectory.new_empty((last.shape[1], 0))
        if extra_rank > 0 and torch.linalg.vector_norm(residual_trajectory) > 1e-8:
            extra, _ = _svd(residual_trajectory.T, extra_rank, method)
            for _ in range(2):
                extra -= reference_device @ (reference_device.T @ extra)
            # Rank-revealing SVD removes directions created by round-off.
            extra, singular, _ = torch.linalg.svd(extra, full_matrices=False)
            extra = extra[:, singular > 1e-5]
        baseline_new = reference_device[:, previous.shape[1]:]
        candidates = torch.cat([baseline_new, extra], dim=1)
        candidates, _ = torch.linalg.qr(candidates, mode="reduced")
        diagnostics["candidate_rank"] = candidates.shape[1]
        old_last = _energy(xlast, old_device)
        target = max(0., required_captured - old_last) / total_energy
        last_factor = residual_last @ candidates / total_energy**.5
        trajectory_factor = trajectory @ candidates
        last_gram = last_factor.T @ last_factor
        trajectory_gram = trajectory_factor.T @ trajectory_factor
        last_gram = (last_gram + last_gram.T) / 2
        trajectory_gram = (trajectory_gram + trajectory_gram.T) / 2
        initial = torch.eye(candidates.shape[1], device=device)[:, :added]
        before = _energy(trajectory, reference_device)
        old_trajectory = _energy(trajectory, old_device)
        best = initial
        best_score = before - old_trajectory
        best_multiplier = None
        best_kind = "last"
        evaluations = 0
        feasible_vectors, infeasible_vectors = initial, None

        def evaluate(multiplier):
            nonlocal best, best_score, best_multiplier, best_kind, evaluations, feasible_vectors, infeasible_vectors
            _, eigenvectors = torch.linalg.eigh(trajectory_gram + multiplier * last_gram)
            vectors = eigenvectors[:, -added:]
            coverage = float((vectors * (last_gram @ vectors)).double().sum())
            score = float((vectors * (trajectory_gram @ vectors)).double().sum())
            evaluations += 1
            feasible = coverage >= target - 1e-7
            if feasible:
                feasible_vectors = vectors
            else:
                infeasible_vectors = vectors
            if feasible and score > best_score + 1e-8:
                best, best_score, best_multiplier = vectors, score, multiplier
                best_kind = "eigenspace"
            return feasible

        if candidates.shape[1] > added:
            if not evaluate(0.):
                low, high = 0., 1.
                for _ in range(32):
                    if evaluate(high):
                        for _ in range(20):
                            middle = (low + high) / 2
                            if evaluate(middle):
                                high = middle
                            else:
                                low = middle
                        break
                    low, high = high, high * 2
        # A Lagrange optimum can jump between eigenspaces at a repeated
        # eigenvalue. A continuous rank-preserving path across the boundary
        # explores feasible compromises rather than silently reverting to last.
        if infeasible_vectors is not None:
            left, _, right = torch.linalg.svd(feasible_vectors.T @ infeasible_vectors)
            aligned = infeasible_vectors @ right.T @ left.T
            low, high = 0., 1.
            for _ in range(24):
                middle = (low + high) / 2
                vectors, _ = torch.linalg.qr((1 - middle) * feasible_vectors + middle * aligned,
                                            mode="reduced")
                coverage = float((vectors * (last_gram @ vectors)).double().sum())
                score = float((vectors * (trajectory_gram @ vectors)).double().sum())
                evaluations += 1
                if coverage >= target - 1e-7:
                    low = middle
                    if score > best_score + 1e-8:
                        best, best_score, best_multiplier = vectors, score, None
                        best_kind = "boundary_mixture"
                else:
                    high = middle
        selected = candidates @ best
        for _ in range(2):
            selected -= old_device @ (old_device.T @ selected)
        selected, _ = torch.linalg.qr(selected, mode="reduced")
        q = torch.cat([old_device, selected], dim=1)
        actual_residual = max(0., total_energy - _energy(xlast, q))
        actual_trajectory = _energy(trajectory, q)
        # Check the actual returned basis, not just reduced Gram estimates.
        accepted = (actual_residual <= allowed_residual + 1e-7 * total_energy
                    and actual_trajectory >= before - 1e-7
                    and q.shape == reference_device.shape)
        if not accepted:
            q, actual_residual, actual_trajectory = reference_device, reference_residual, before
        diagnostics.update(
            selection="trajectory" if accepted and best_kind != "last" else "last_fallback",
            solver_candidate=best_kind if accepted else "last",
            solver_evaluations=evaluations, solver_multiplier=best_multiplier,
            trajectory_coverage_before=before, trajectory_coverage_after=actual_trajectory,
            last_residual_fraction=actual_residual / total_energy,
            relative_residual_after_update=actual_residual / total_energy,
            floor_satisfied=actual_residual <= allowed_residual + 1e-7 * total_energy,
        )
        return q.cpu().contiguous(), diagnostics


class TrajectoryBasisStore:
    def __init__(self, svd_method="randomized", floor="tolerance"):
        if svd_method not in {"randomized", "exact"} or floor not in {"tolerance", "reference"}:
            raise ValueError("Invalid trajectory selection configuration")
        self.bases = {}
        self.svd_method, self.floor = svd_method, floor

    def update(self, activations, max_rank, residual_tolerance, device="cpu"):
        grouped = {}
        for key, rows in activations.items():
            module, depth = key.rsplit("@ut", 1)
            grouped.setdefault(module, {})[int(depth)] = rows
        if not grouped or (self.bases and set(grouped) != set(self.bases)):
            raise ValueError("Every task must provide the same trajectory modules")
        reports, next_bases = [], {}
        for module, depths in sorted(grouped.items()):
            if sorted(depths) != list(range(len(depths))) or len(depths) < 2:
                raise ValueError(f"{module}: missing trajectory depths")
            started = time.perf_counter()
            xs = [depths[d] for d in sorted(depths)]
            q, report = select_trajectory(xs, self.bases.get(module), max_rank, residual_tolerance,
                device=device, method=self.svd_method, floor=self.floor, head=not module.startswith("layers."))
            next_bases[module] = q
            reports.append({"layer": module, "source_depths": sorted(depths),
                "sampled_vectors": sum(x.shape[0] for x in xs),
                "paired_positions": xs[-1].shape[0], "selection_seconds": time.perf_counter() - started,
                **report})
        self.bases = next_bases
        return reports

    def storage_bytes(self):
        return sum(q.numel() * q.element_size() for q in self.bases.values())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": "trajectory_v1", "bases": self.bases,
                    "svd_method": self.svd_method, "floor": self.floor}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != "trajectory_v1":
            raise ValueError("Unsupported trajectory history checkpoint")
        store = cls(state["svd_method"], state["floor"])
        store.bases = state["bases"]
        return store
