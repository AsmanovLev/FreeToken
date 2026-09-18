"""CPU tests for the learned expert-usage planner (moe/expert_usage.py)."""

from __future__ import annotations

import torch

from freetoken.moe.expert_usage import (
    load_usage,
    merge_usage,
    plan_hot_set,
    plan_is_identity,
    save_usage,
    usage_path,
)

L, E = 3, 8


def test_plan_puts_hot_experts_in_the_prefix():
    freq = torch.zeros(L, E, dtype=torch.int64)
    freq[:, 5] = 100
    freq[:, 2] = 50
    freq[:, 0] = 10
    perm, perm_inv, hit = plan_hot_set(freq, ram_experts=2)
    # hot experts (5, then 2) land in rows 0 and 1 for every layer
    assert perm_inv[:, 0].tolist() == [5, 5, 5]
    assert perm_inv[:, 1].tolist() == [2, 2, 2]
    assert perm_inv[:, 2].tolist() == [0, 0, 0]
    # perm is the inverse of perm_inv
    rows = torch.arange(E).expand(L, E)
    assert torch.equal(perm.gather(1, perm_inv), rows)
    # hit = (100 + 50) / 160
    assert abs(hit - 150 / 160) < 1e-9
    assert not plan_is_identity(perm)


def test_plan_uniform_is_identity_with_uniform_hit():
    freq = torch.ones(L, E, dtype=torch.int64)
    perm, perm_inv, hit = plan_hot_set(freq, ram_experts=4)
    assert plan_is_identity(perm)
    assert torch.equal(perm_inv, torch.arange(E).expand(L, E))
    assert abs(hit - 0.5) < 1e-9


def test_plan_rejects_a_bad_budget():
    freq = torch.ones(L, E, dtype=torch.int64)
    for bad in (0, E + 1, -1):
        try:
            plan_hot_set(freq, ram_experts=bad)
        except ValueError:
            continue
        raise AssertionError(f"ram_experts={bad} should raise")


def test_usage_roundtrip_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EXPERT_USAGE_DIR", str(tmp_path))
    freq = torch.arange(L * E, dtype=torch.int64).reshape(L, E)
    path = usage_path(str(tmp_path / "model"), L, E)
    assert load_usage(path, L, E) is None
    save_usage(path, freq, str(tmp_path / "model"))
    got = load_usage(path, L, E)
    assert got is not None and torch.equal(got, freq)
    merged = merge_usage(got, freq)
    assert torch.equal(merged, freq * 2)
    # geometry mismatch is rejected, not reshaped
    assert load_usage(path, L, E + 1) is None


def test_usage_path_is_stable_and_model_specific(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_EXPERT_USAGE_DIR", str(tmp_path))
    a = usage_path("/models/a", L, E)
    b = usage_path("/models/b", L, E)
    assert a == usage_path("/models/a", L, E)
    assert a != b
