"""Shared linear-subspace utilities for full-parameter history protection."""
from __future__ import annotations

import torch


def depth_basis_key(layer, depth):
    return f"{layer}@ut{depth}"


def project_activations(x, basis):
    if basis is None or not basis.numel():
        return x
    basis = basis.to(x)
    return x - (x @ basis) @ basis.T


def orthogonal_span(matrix):
    """Rank-revealing span; dependent columns must not add constraints."""
    if not matrix.numel():
        return matrix[:, :0]
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * s[0]
    return u[:, s > tolerance].contiguous()


def union_basis(bases, layer_key, depth=None):
    """Build the union of a module's depth-specific historical banks."""
    prefix = f"{layer_key}@ut"
    selected = [v.float().cpu() for k, v in bases.items()
                if (k.startswith(prefix) and (depth is None or k == f"{layer_key}@ut{depth}"))
                and v.numel()]
    return orthogonal_span(torch.cat(selected, dim=1)) if selected else None
