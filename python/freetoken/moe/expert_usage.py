"""Learned expert-usage statistics for the NVMe disk tier (Colibri-style hot-set).

The disk tier pins a contiguous RAM prefix ``[0, ram_experts)`` per bank (see
``moe.disk_tier``). Which experts land in that prefix is normally just the
checkpoint order -- fine when routing is uniform, wasteful when it is skewed.
This module learns the *routed* frequency per (layer, expert) across sessions
and turns it into a row permutation: hot experts get small row ids, so the
existing prefix pin/release machinery serves the hot set without a kernel
change.

Everything here is plain torch/int arithmetic (no GPU): the planner is unit
testable and the loader/disk-tier only consume the produced permutation.

Usage file layout (``*.pt``):
    {
      "version": 1,
      "model": str,            # opaque id (model path + expert geometry)
      "experts": int,          # E
      "layers": int,           # L (MoE layers)
      "freq": LongTensor[L, E],# routed counts (decode + prefill)
    }
"""

from __future__ import annotations

import hashlib
import os

import torch

_VERSION = 1


def usage_dir() -> str:
    return os.environ.get(
        "FT_EXPERT_USAGE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "freetoken", "expert_usage"),
    )


def usage_path(model_path: str, num_layers: int, num_experts: int) -> str:
    """Stable per-model usage file: path + geometry hashed so two models never mix."""
    key = f"{os.path.abspath(model_path)}|{num_layers}|{num_experts}".encode()
    tag = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join(usage_dir(), f"{tag}.pt")


def load_usage(path: str, num_layers: int, num_experts: int) -> torch.Tensor | None:
    """[L, E] routed counts, or None when the file is missing/incompatible."""
    if not os.path.isfile(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(obj, dict) or obj.get("version") != _VERSION:
        return None
    freq = obj.get("freq")
    if not isinstance(freq, torch.Tensor) or tuple(freq.shape) != (num_layers, num_experts):
        return None
    return freq.to(torch.int64)


def save_usage(path: str, freq: torch.Tensor, model_path: str) -> None:
    """Atomically persist [L, E] counts; never fail the server on a write error."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layers, experts = freq.shape
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {
            "version": _VERSION,
            "model": os.path.abspath(model_path),
            "experts": int(experts),
            "layers": int(layers),
            "freq": freq.detach().to(torch.int64).cpu(),
        },
        tmp,
    )
    os.replace(tmp, path)


def merge_usage(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    """Accumulate this run's counts into the persisted history."""
    if old is None:
        return new.detach().to(torch.int64).cpu()
    return (old.to(torch.int64) + new.detach().to(torch.int64).cpu())


def plan_hot_set(
    freq: torch.Tensor, ram_experts: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Rank experts per layer by routed frequency.

    Returns ``(perm, perm_inv, expected_hit)`` where ``perm[layer, e]`` is the
    bank row expert ``e`` should live in (hot -> small row id; ties broken by
    expert id) and ``perm_inv`` the inverse. ``expected_hit`` is the fraction of
    routed events served by rows ``[0, ram_experts)`` under this ordering
    (the training-set hit; the disk tier's own oracle is its upper bound).
    """
    layers, experts = freq.shape
    if not 0 < ram_experts <= experts:
        raise ValueError(f"ram_experts must be in (1, {experts}], got {ram_experts}")
    f = freq.to(torch.int64)
    # descending frequency, ascending expert id on ties: a stable, reproducible order
    order = torch.argsort(f, dim=1, descending=True, stable=True)
    perm_inv = order  # row r -> expert
    perm = torch.empty_like(perm_inv)
    perm.scatter_(1, perm_inv, torch.arange(experts).expand(layers, experts))
    total = f.sum().clamp(min=1)
    hit = f.gather(1, perm_inv[:, :ram_experts]).sum() / total
    return perm, perm_inv, float(hit)


def plan_is_identity(perm: torch.Tensor) -> bool:
    """True when the permutation is the identity (no learned reordering needed)."""
    experts = perm.shape[1]
    return bool(torch.equal(perm, torch.arange(experts).expand_as(perm)))
