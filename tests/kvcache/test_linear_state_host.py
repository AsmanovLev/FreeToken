"""LinearStatePool host snapshot store tests (snapshot_mode="host").

CPU roundtrip is exact; the CUDA variant exercises the engine-stream ordering
(D2H enqueued + event-synced, H2D on restore). No kernels involved.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    _linear_pool_num_slots,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


def _group():
    return LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1),
        num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )


def _pool(num_slots=4, device="cpu", mode="host", slot_states=(SlotStateSpec("m", (2,), fill_value=0.5),)):
    return LinearStatePool(
        group=_group(), num_slots=num_slots, dtype=torch.bfloat16,
        device=torch.device(device), tp_size=1, slot_states=slot_states,
        snapshot_mode=mode,
    )


def _fill(pool, slot: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    c = pool.conv_states[:, slot]
    pool.conv_states[:, slot] = torch.randn(
        c.shape, generator=g
    ).to(c.dtype)
    r = pool.recurrent_states[:, slot]
    pool.recurrent_states[:, slot] = torch.randn(
        r.shape, generator=g
    ).to(r.dtype)
    pool.slot_states["m"][:, slot] = 1.25


def _snapshot(pool, slot: int):
    return (
        pool.conv_states[:, slot].clone(),
        pool.recurrent_states[:, slot].clone(),
        pool.slot_states["m"][:, slot].clone(),
    )


def _assert_equal(pool, slot: int, snap) -> None:
    assert torch.equal(pool.conv_states[:, slot], snap[0])
    assert torch.equal(pool.recurrent_states[:, slot], snap[1])
    assert torch.equal(pool.slot_states["m"][:, slot], snap[2])


def test_host_roundtrip_cpu():
    pool = _pool(num_slots=4, device="cpu")
    _fill(pool, 1, seed=3)
    snap = _snapshot(pool, 1)

    host_id = pool.snapshot_to_host(1)
    assert pool.is_host_snapshot(host_id)
    assert pool.num_host_snapshots == 1
    assert pool.host_snapshot_bytes() > 0

    # clobber the live slot; the tree snapshot must be unaffected
    pool.clear_slots([1])
    assert not torch.equal(pool.conv_states[:, 1], snap[0])

    # COW-restore into a fresh live slot through the copy_from routing
    live = pool.alloc(1)[0]
    pool.copy_from(host_id, live)
    _assert_equal(pool, live, snap)

    # VRAM free-list untouched by host free
    pool.free([host_id])
    assert pool.num_host_snapshots == 0
    assert pool.num_free_slots == 2  # host free must not touch the VRAM free-list


def test_host_reuse_and_double_free():
    pool = _pool(num_slots=4, device="cpu")
    _fill(pool, 1, seed=5)
    h1 = pool.snapshot_to_host(1)
    pool.free([h1])
    # reuse returns the same entry index; new data fully overwrites
    _fill(pool, 2, seed=7)
    snap2 = _snapshot(pool, 2)
    h2 = pool.snapshot_to_host(2)
    assert h2 == h1
    pool.clear_slots([2])
    live = pool.alloc(1)[0]
    pool.copy_from(h2, live)
    _assert_equal(pool, live, snap2)
    pool.free([h2])
    with pytest.raises(ValueError):
        pool.free([h2])  # double free
    with pytest.raises(ValueError):
        pool.free([h2 + 100])  # out of range


def test_gpu_mode_untouched():
    pool = _pool(num_slots=4, device="cpu", mode="gpu")
    assert not pool.host_snapshots
    _fill(pool, 1, seed=9)
    snap = _snapshot(pool, 1)
    live = pool.alloc(1)[0]
    pool.copy_from(1, live)  # plain D2D still works
    _assert_equal(pool, live, snap)


def test_num_slots_formula():
    class C:
        max_running_req = 1
        cache_type = "hybrid_radix"
        linear_state_cache_ratio = 2.0

    c = C()
    assert _linear_pool_num_slots(c) == 4 * 1 + 4 + 1
    c.linear_state_snapshots = "host"
    assert _linear_pool_num_slots(c) == 2 * 1 + 2
    c.cache_type = "naive"
    assert _linear_pool_num_slots(c) == 1 + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_host_roundtrip_cuda():
    pool = _pool(num_slots=4, device="cuda")
    stream = torch.cuda.Stream()
    pool.set_engine_stream(stream)
    _fill(pool, 1, seed=11)
    snap = _snapshot(pool, 1)

    host_id = pool.snapshot_to_host(1)  # cross-stream: enqueue + event sync
    pool.clear_slots([1])
    live = pool.alloc(1)[0]
    pool.copy_from(host_id, live)
    torch.cuda.synchronize()
    _assert_equal(pool, live, snap)
    pool.free([host_id])
    assert pool.num_host_snapshots == 0
