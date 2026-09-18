"""CPU tests for the EXL3 disk tier (moe/disk_tier.py).

A synthetic EXL3 (QTIP trellis) MoE checkpoint (2 layers, mixed-rate K
recipe, 4 experts) is written as safetensors shards with deterministic
per-tensor content. The tests verify that :class:`Exl3DiskIndex` resolves
byte ranges that RECONSTRUCT a packed bank row byte-for-byte (zero padding
of the mixed-rate trellis fields included), that the per-layer K list
matches the checkpoint recipe, and that :class:`DiskTier`'s exl3 fetch
path places the right bytes into slot-cache rows and rewrites the miss
list. No CUDA needed -- the "GPU" banks here are CPU tensors and the
staging pin is stubbed.
"""

from __future__ import annotations

import json
import struct
import threading
import types

import pytest
import torch

from freetoken.moe.disk_tier import DiskTier, Exl3DiskIndex
from freetoken.moe.host_banks import HostBank

H, I, E, L = 64, 32, 4, 2
DENSE = 1  # one leading dense layer: checkpoint hf layers are DENSE + [0, L)
KS = (3, 4)  # mixed-rate recipe: layer 0 K=3, layer 1 K=4
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def _name(layer, expert, proj, kind):
    return f"model.language_model.layers.{DENSE + layer}.mlp.experts.{expert}.{proj}.{kind}"


def _tensor_spec(layer, proj, kind):
    """(shape, dtype) of one (layer, proj, kind) checkpoint tensor."""
    k = KS[layer]
    if kind == "trellis":
        if proj == "down_proj":
            return (I // 16, H // 16, k * 16), torch.int16
        return (H // 16, I // 16, k * 16), torch.int16
    if proj == "down_proj":
        return ((I,) if kind == "suh" else (H,)), torch.float16
    return ((H,) if kind == "suh" else (I,)), torch.float16


def _tensor_bytes(layer, proj, kind):
    """Packed byte count of the checkpoint tensor (tb for trellis)."""
    shape, dtype = _tensor_spec(layer, proj, kind)
    n = 1
    for d in shape:
        n *= d
    return n * (2 if dtype == torch.int16 else 2)


def _tensor_for(layer, expert, proj, kind):
    shape, dtype = _tensor_spec(layer, proj, kind)
    proj_i = ("gate_proj", "up_proj", "down_proj").index(proj)
    kind_i = ("trellis", "suh", "svh").index(kind)
    base = layer * 100000 + expert * 1000 + proj_i * 100 + kind_i * 10
    n = 1
    for d in shape:
        n *= d
    if dtype == torch.int16:
        return torch.arange(n, dtype=torch.int32).add_(base % 32768).to(torch.int16).reshape(shape)
    return (torch.arange(n, dtype=torch.float32).add_(base % 32767) / 9).to(torch.float16).reshape(shape)


def _bank_row(layer, expert):
    """(gate_up row, down row) flat uint8 exactly as _setup_exl3_banks packs it."""
    tb = H * I * KS[layer] // 8
    tb_max = H * I * max(KS) // 8
    off = 0

    def put(buf, t):
        nonlocal off
        b = t.view(torch.uint8).reshape(-1)
        buf[off:off + b.numel()] = b
        off += b.numel()

    def pad(buf):
        nonlocal off
        buf[off:off + tb_max - tb] = 0
        off += tb_max - tb

    gu = torch.zeros(2 * tb_max + 4 * H + 4 * I, dtype=torch.uint8)
    for proj in ("gate_proj", "up_proj"):
        put(gu, _tensor_for(layer, expert, proj, "trellis"))
        pad(gu)
    for proj in ("gate_proj", "up_proj"):  # gate suh, gate svh, up suh, up svh
        put(gu, _tensor_for(layer, expert, proj, "suh"))
        put(gu, _tensor_for(layer, expert, proj, "svh"))
    assert off == gu.numel()
    off = 0
    dn = torch.zeros(tb_max + 2 * H + 2 * I, dtype=torch.uint8)
    put(dn, _tensor_for(layer, expert, "down_proj", "trellis"))
    pad(dn)
    put(dn, _tensor_for(layer, expert, "down_proj", "suh"))
    put(dn, _tensor_for(layer, expert, "down_proj", "svh"))
    assert off == dn.numel()
    return gu, dn


@pytest.fixture()
def checkpoint(tmp_path):
    import safetensors.torch

    by_shard = {s: {} for s in SHARDS}
    weight_map = {}
    for layer in range(L):
        for expert in range(E):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("trellis", "suh", "svh"):
                    shard = SHARDS[(layer * E + expert) % 2]
                    name = _name(layer, expert, proj, kind)
                    by_shard[shard][name] = _tensor_for(layer, expert, proj, kind)
                    weight_map[name] = shard
    for shard, tensors in by_shard.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard), metadata={"format": "pt"})
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    config = types.SimpleNamespace(num_experts=E, num_moe_layers=L, num_layers=DENSE + L,
                                   hidden_size=H, moe_intermediate_size=I)
    return tmp_path, config


def _index(checkpoint):
    path, config = checkpoint
    return Exl3DiskIndex(str(path), config)


def test_index_k_and_row_bytes_match_the_recipe(checkpoint):
    index = _index(checkpoint)
    assert index.k_per_layer == KS
    tb_max = H * I * max(KS) // 8
    assert index.row_bytes["gate_up"] == 2 * tb_max + 4 * H + 4 * I
    assert index.row_bytes["down"] == tb_max + 2 * H + 2 * I
    assert len(index.shard_paths) == 2


def test_index_segments_resolve_the_checkpoint_bytes(checkpoint):
    path, _config = checkpoint
    index = _index(checkpoint)
    headers = {}
    for shard in SHARDS:
        with open(path / shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            headers[shard] = json.loads(f.read(hlen))
    for layer in range(L):
        for expert in range(E):
            gu, dn = _bank_row(layer, expert)
            for bank_idx, (row, expected) in enumerate([(gu, "gate_up"), (dn, "down")]):
                got = torch.zeros(row.numel(), dtype=torch.uint8)
                for (shard_idx, off, nbytes, dst_off, field) in \
                        index.row_segments(bank_idx, layer, expert):
                    shard_path = path / SHARDS[shard_idx]
                    with open(shard_path, "rb") as f:
                        f.seek(off)
                        data = f.read(nbytes)
                    assert len(data) == nbytes
                    got[dst_off:dst_off + nbytes] = torch.frombuffer(
                        bytearray(data), dtype=torch.uint8)
                assert torch.equal(got.cpu(), row.cpu()), (expected, layer, expert)


def _fake_cache(tb_max):
    gu_row = 2 * tb_max + 4 * H + 4 * I
    dn_row = tb_max + 2 * H + 2 * I
    banks = [
        (
            [torch.zeros(E, gu_row, dtype=torch.uint8) for _ in range(L)],
            torch.full((8, gu_row), 0xFF, dtype=torch.uint8),
        ),
        (
            [torch.zeros(E, dn_row, dtype=torch.uint8) for _ in range(L)],
            torch.full((8, dn_row), 0xFF, dtype=torch.uint8),
        ),
    ]
    cache = type("FakeCache", (), {})()
    cache.banks = banks
    cache.num_experts = E
    cache.num_layers = L
    cache.num_indices = torch.tensor([0], dtype=torch.int64)
    cache.src_indices = torch.zeros(64, dtype=torch.int32)
    cache.evict_slots = torch.zeros(64, dtype=torch.int32)
    return cache


def _tier(checkpoint, cache, ram_experts=2, perm=None):
    index = _index(checkpoint)
    tier = DiskTier(index, cache, ram_experts=ram_experts, workers=2, perm=perm)
    max_row = max(index.row_bytes.values())
    assert tier._staging_size >= max_row + 2 * 4096, (tier._staging_size, max_row)
    local = threading.local()

    def _staging_ring():
        ring = getattr(local, "ring", None)
        if ring is None:
            ring = [[HostBank((tier._staging_size,), torch.uint8), None]
                    for _ in range(tier._STAGING_RING)]
            local.ring = ring
        return ring

    tier._staging_ring = _staging_ring
    return tier


def test_fetch_expert_places_packed_rows(checkpoint):
    cache = _fake_cache(_index(checkpoint).tb_max)
    tier = _tier(checkpoint, cache)
    for layer in range(L):
        for expert in range(E):
            slot = (layer * E + expert) % 8
            tier._fetch_expert(layer, expert, slot)
            gu, dn = _bank_row(layer, expert)
            for bank_idx, expected in enumerate((gu, dn)):
                (_host, gpu_cache) = cache.banks[bank_idx]
                got = gpu_cache[slot].contiguous().view(torch.uint8).reshape(-1)
                assert torch.equal(got, expected), (bank_idx, layer, expert)
    stats = tier.stats()
    assert stats["experts_fetched"] == L * E


def test_fetch_pending_filters_and_rewrites(checkpoint):
    cache = _fake_cache(_index(checkpoint).tb_max)
    tier = _tier(checkpoint, cache, ram_experts=2)  # experts 0,1 RAM; 2,3 disk
    layer = 1
    cache.src_indices[:3] = torch.tensor([0, 2, 3], dtype=torch.int32)
    cache.evict_slots[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)
    cache.num_indices.fill_(3)

    tier.fetch_pending(cache, layer)

    gu, dn = _bank_row(layer, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(gpu_cache[7].contiguous().view(torch.uint8).reshape(-1),
                           (gu, dn)[bank_idx])
    # ...and the miss list shrank to the RAM-resident remainder.
    assert cache.num_indices.item() == 1
    assert cache.src_indices[0].item() == 0
    assert cache.evict_slots[0].item() == 5


def test_fetch_pending_all_ram_is_noop(checkpoint):
    cache = _fake_cache(_index(checkpoint).tb_max)
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.num_indices.fill_(2)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 2
    assert tier.stats()["experts_fetched"] == 0


def test_fetch_pending_all_disk_clears_list(checkpoint):
    cache = _fake_cache(_index(checkpoint).tb_max)
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:1] = torch.tensor([3], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([4], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 0
    gu, dn = _bank_row(0, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(gpu_cache[4].contiguous().view(torch.uint8).reshape(-1),
                           (gu, dn)[bank_idx])


def test_fetch_pending_translates_permuted_rows(checkpoint):
    """A learned row permutation: the cache hands ROW ids, the tier fetches the
    checkpoint EXPERT those rows hold (perm_inv) into the same slots."""
    cache = _fake_cache(_index(checkpoint).tb_max)
    perm = torch.tensor([[1, 2, 0, 3]] * L, dtype=torch.int64)  # expert -> row
    tier = _tier(checkpoint, cache, ram_experts=2, perm=perm)
    layer = 1
    # row 2 holds expert perm_inv[2] == 1; row 3 holds expert 3
    cache.src_indices[:2] = torch.tensor([2, 3], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([5, 6], dtype=torch.int32)
    cache.num_indices.fill_(2)

    tier.fetch_pending(cache, layer)

    gu1, dn1 = _bank_row(layer, 1)
    gu3, dn3 = _bank_row(layer, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(gpu_cache[5].contiguous().view(torch.uint8).reshape(-1),
                           (gu1, dn1)[bank_idx])
        assert torch.equal(gpu_cache[6].contiguous().view(torch.uint8).reshape(-1),
                           (gu3, dn3)[bank_idx])
    assert cache.num_indices.item() == 0


def test_remap_routing_to_rows_uses_the_layer_perm():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = types.SimpleNamespace(hot_perm=torch.tensor([[1, 2, 0, 3]]))
    ids = torch.tensor([[0, 3, 2]], dtype=torch.int32)
    OffloadMoeCache.remap_routing_to_rows(cache, 0, ids)
    assert ids.tolist() == [[1, 3, 0]]
    cache.hot_perm = None
    OffloadMoeCache.remap_routing_to_rows(cache, 0, ids)
    assert ids.tolist() == [[1, 3, 0]]  # no-op without a permutation


def test_prefetch_serves_the_next_pass_from_host(checkpoint, monkeypatch):
    """Temporal prefetch: the rows a layer misses are pre-read into host buffers,
    and the next pass's demand fetch takes them as an H2D copy (no disk)."""
    import time

    monkeypatch.setenv("FT_DISK_PREFETCH_ROWS", "8")
    cache = _fake_cache(_index(checkpoint).tb_max)
    tier = _tier(checkpoint, cache, ram_experts=2)
    layer = 1
    rows = [2, 3]

    def demand():
        cache.src_indices[:2] = torch.tensor(rows, dtype=torch.int32)
        cache.evict_slots[:2] = torch.tensor([5, 6], dtype=torch.int32)
        cache.num_indices.fill_(2)
        tier.fetch_pending(cache, layer)

    demand()
    deadline = time.time() + 10
    while time.time() < deadline:
        with tier._pf_lock:
            if not tier._pf_inflight and tier._pf:
                break
        time.sleep(0.01)
    assert tier.stats()["prefetch_fetches"] >= 1

    # wipe the slots, then a second pass for the same rows must use the prefetch
    for _h, gpu in cache.banks:
        gpu[5].fill_(0xFF)
        gpu[6].fill_(0xFF)
    hits_before = tier.stats()["prefetch_hits"]
    demand()
    assert tier.stats()["prefetch_hits"] == hits_before + 2
    gu2, dn2 = _bank_row(layer, 2)
    gu3, dn3 = _bank_row(layer, 3)
    for bank_idx, (_host, gpu) in enumerate(cache.banks):
        assert torch.equal(gpu[5].view(torch.uint8).reshape(-1), (gu2, dn2)[bank_idx])
        assert torch.equal(gpu[6].view(torch.uint8).reshape(-1), (gu3, dn3)[bank_idx])


def test_loader_places_experts_in_permuted_rows(checkpoint):
    """_setup_exl3_banks(hot_perm=...) writes expert e into bank row perm[layer,e]
    and leaves disk rows (>= ram_experts) unbacked."""
    from freetoken.distributed import set_tp_info
    from freetoken.models.qwen3_5_moe.weight import _setup_exl3_banks

    path, config = checkpoint
    try:
        set_tp_info(rank=0, size=1)
    except Exception:
        pass
    perm = torch.tensor([[1, 2, 0, 3]] * L, dtype=torch.int64)  # expert -> row
    banks = _setup_exl3_banks(str(path), config, torch.device("cpu"), False,
                              ram_experts=2, hot_perm=perm)
    for layer in range(L):
        gu_rows = banks.sources["gate_up"][layer]
        dn_rows = banks.sources["down"][layer]
        gu_expected = {_row_id: _bank_row(layer, e)[0] for e, _row_id in
                       enumerate(perm[layer].tolist()) if _row_id < 2}
        dn_expected = {_row_id: _bank_row(layer, e)[1] for e, _row_id in
                       enumerate(perm[layer].tolist()) if _row_id < 2}
        for r in range(E):
            got_gu = gu_rows[r].view(torch.uint8).reshape(-1)
            got_dn = dn_rows[r].view(torch.uint8).reshape(-1)
            if r in gu_expected:
                assert torch.equal(got_gu, gu_expected[r].view(torch.uint8).reshape(-1)), \
                    ("gate_up", layer, r)
                assert torch.equal(got_dn, dn_expected[r].view(torch.uint8).reshape(-1)), \
                    ("down", layer, r)
            else:
                assert not got_gu.any() and not got_dn.any(), ("unbacked", layer, r)


def test_disk_index_ignores_non_expert_keys(tmp_path):
    """Routed-expert anchors only: MTP / shared-expert / eigen tensors must be
    ignored by the key match (else an exl3 checkpoint with extra tables would
    double-map a row)."""
    import safetensors.torch

    weight_map = {"model.mtp.expert_gate_proj": "a.safetensors"}
    import types

    config = types.SimpleNamespace(num_experts=E, num_moe_layers=L, num_layers=DENSE + L,
                                   hidden_size=H, moe_intermediate_size=I)
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    with pytest.raises(KeyError):
        _index((tmp_path, config))
