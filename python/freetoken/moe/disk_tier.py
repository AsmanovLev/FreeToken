"""Disk tier: NVMe-backed MoE experts (VRAM <- RAM <- NVMe).

Lets the offload backend serve experts that do NOT fit in pinned RAM: the RAM
bank holds only the first ``ram_experts`` experts per layer (pinned), the rest
stay on disk in the original checkpoint. When the GPU slot cache misses a
disk-resident expert, :class:`DiskTier` fetches its rows with O_DIRECT preadv
into a small pinned staging buffer and H2D-copies them into the slot the LRU
kernel already assigned, then shrinks the miss list so the existing PCIe
``copy_missing`` path only moves the RAM-resident misses.

v0 scope (prototype):
* native NVFP4 layout only (the "triton" backend banks -- what sm_120 picks);
* ``decode_target == "gpu"`` (offload) only -- the CPU executor reads banks
  directly and would read released pages;
* synchronous fetch (the layer waits for its disk misses); no CUDA-graph
  capture (the miss-list D2H/H2D round trip is host-side and variable);
* prefill_overlap off (the double-buffer prefill path bypasses the slot cache).

The bank rows are read from the ORIGINAL safetensors shards: every expert
tensor is a contiguous per-expert tensor, so a bank row is one (or two, for
the gate|up-fused banks) aligned super-block preads. No FTW conversion needed.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank

_ALIGN = 4096


@dataclass(frozen=True)
class DiskTierSpec:
    """Engine -> loader: how many experts per layer stay pinned in RAM.

    Experts ``[0, ram_experts)`` are pinned as usual; ``[ram_experts, E)`` keep
    their bank rows allocated but their pages are released after load and are
    served from disk by :class:`DiskTier`.

    ``perm`` (optional, ``[L, E]`` int tensor) is the learned hot-set row order:
    ``perm[layer, expert]`` is the bank row that expert's weights live in, with
    hot experts mapped to the low rows so the pinned prefix serves them. When
    None the identity order is used (checkpoint order)."""

    ram_experts: int
    perm: "torch.Tensor | None" = None


def release_bank_tails(banks_by_name: dict[str, list[HostBank]], num_experts: int,
                       ram_experts: int) -> None:
    """MADV_DONTNEED the unpinned tail rows of every bank layer (post-load).

    The release is an optimization, not an invariant: the tail rows were never
    written at load, so when a row boundary is not page-aligned (the small scale
    banks) we warn and skip that bank instead of failing the boot."""
    _PAGE = 4096
    for layer_banks in banks_by_name.values():
        for bank in layer_banks:
            row_bytes = bank.nbytes // num_experts
            offset = ram_experts * row_bytes
            size = bank.nbytes - offset
            if offset % _PAGE or size % _PAGE:
                print(f"[disk-tier] WARNING: bank row boundary not page-aligned "
                      f"(ram_experts={ram_experts}, row_bytes={row_bytes}); skipping "
                      f"the release for this bank -- the tail rows were never written, "
                      f"so nothing is lost", flush=True)
                continue
            bank.release_range(offset, size)


def tail_resident_bytes(bank: HostBank, num_experts: int, ram_experts: int) -> int:
    """Bytes the kernel currently backs in the released tail rows ``[ram_experts, E)``.

    mincore(2) over the tail's byte range: one syscall, per-page residency.
    Conservative -- mincore also reports private-anon pages mapped from the
    shared zero page (a plain READ of a tail row), so this overcounts what
    actually costs RAM (cgroup memory.stat shmem is the real number).
    Returns -1 if mincore itself fails."""
    row_bytes = bank.nbytes // num_experts
    off = ram_experts * row_bytes
    size = bank.nbytes - off
    if size <= 0:
        return 0
    _PAGE = 4096
    vec = (ctypes.c_ubyte * (size // _PAGE))()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    if libc.mincore(ctypes.c_void_p(bank.addr + off), size, vec) != 0:
        return -1
    return sum(vec) * _PAGE


def check_tail_unbacked(banks_by_name: dict[str, list[HostBank]], num_experts: int,
                        ram_experts: int) -> None:
    """Startup sanity check for the lazy-tail invariant, right after
    ``release_bank_tails``: nothing reads or writes the tail rows, so the kernel
    should be backing ~none of them. The expected worst case is one 2 MiB THP
    huge page per bank layer (shmem_enabled=always|force can back the
    prefix/tail boundary as a huge page); more than that means something touched
    the tail and the disk-tier RAM math no longer holds. Always logs, warns
    above the bound."""
    _HUGE = 2 << 20
    resident = tail = n_banks = 0
    for layer_banks in banks_by_name.values():
        for bank in layer_banks:
            row_bytes = bank.nbytes // num_experts
            t = bank.nbytes - ram_experts * row_bytes
            if t <= 0:
                continue
            r = tail_resident_bytes(bank, num_experts, ram_experts)
            if r < 0:
                continue  # mincore failed: skip rather than warn on our own probe
            resident += r
            tail += t
            n_banks += 1
    if n_banks == 0:
        return
    from freetoken.distributed import try_get_tp_info
    tp = try_get_tp_info()
    rank = getattr(tp, "rank", "?")
    size_ = getattr(tp, "size", "?")
    bound = n_banks * _HUGE
    print(f"[disk-tier] tail check rank={rank}/{size_}: resident {resident >> 20} MiB "
          f"of {tail >> 20} MiB (warn bound {bound >> 20} MiB = 1x2MiB per bank layer)",
          flush=True)
    if resident > bound:
        print(f"[disk-tier] WARNING: tail rows more resident than the THP bound -- "
              f"something is reading the released rows; the disk-tier RAM math no longer holds",
              flush=True)

# Native NVFP4 bank order (== _BANK_SCHEMAS["nvfp4"]) and, per bank, the
# checkpoint segments that make up one expert row: (proj, kind, dst_row_start,
# dst_row_end). The gate|up-fused banks splice gate rows then up rows on the
# output-row axis; down banks are a single segment. Row ends are None = rest.
_NVP4_BANK_SEGS = (
    (("gate_proj", "weight", 0, None), ("up_proj", "weight", None, None)),
    (("gate_proj", "weight_scale", 0, None), ("up_proj", "weight_scale", None, None)),
    (("gate_proj", "weight_scale_2", 0, None), ("up_proj", "weight_scale_2", None, None)),
    (("down_proj", "weight", 0, None),),
    (("down_proj", "weight_scale", 0, None),),
    (("down_proj", "weight_scale_2", 0, None),),
)

# EXL3 bank order (== _BANK_SCHEMAS["exl3"]). Row fields are spliced on the
# BYTE axis (see moe/fused_exl3.py for the layout doc), so segments carry their
# own absolute row-destination offset instead of the nvfp4 output-row split.
_EXL3_BANK_SEGS = (
    # gate_up: gate.trellis @0 | up.trellis @tb_max | gate.suh | gate.svh | up.suh | up.svh
    (
        ("gate_proj", "trellis", 0),
        ("up_proj", "trellis", "tb_max"),
        ("gate_proj", "suh", "2_tb_max"),
        ("gate_proj", "svh", "2_tb_max+2H"),
        ("up_proj", "suh", "2_tb_max+2H+2I"),
        ("up_proj", "svh", "2_tb_max+4H+2I"),
    ),
    # down: down.trellis @0 | down.suh @tb_max | down.svh @tb_max+2I
    (
        ("down_proj", "trellis", 0),
        ("down_proj", "suh", "tb_max"),
        ("down_proj", "svh", "tb_max+2I"),
    ),
)

_EXL3_FIELD_VAR = {
    "0": "zero", "tb_max": "tb_max", "2_tb_max": "2*tb_max", "2H": "2*H", "2I": "2*I",
}



def _preadv_error(tier, staging, shard_idx: int, off: int, a0: int, slen: int,
                  direct: bool) -> OSError:
    vma = "?"
    try:
        for line in open("/proc/self/maps"):
            lo, hi = line.split()[0].split("-")
            if int(lo, 16) <= staging.addr < int(hi, 16):
                vma = line.strip()[:120]
                break
    except OSError:
        pass
    return OSError(
        f"disk-tier preadv failed: shard={shard_idx} off={off} a0={a0} "
        f"slen={slen} direct={direct} buf={hex(staging.addr)} "
        f"staging_size={tier._staging_size} thread={threading.current_thread().name} "
        f"vma={vma}"
    )


def _read_safetensors_offsets(path: str) -> dict[str, tuple[int, int]]:
    """{tensor_name: (start, end)} from a shard's safetensors header, as ABSOLUTE
    file offsets (data_offsets are relative to the data section, i.e. after the
    8-byte length + header JSON)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
    base = 8 + hlen
    return {
        k: (v["data_offsets"][0] + base, v["data_offsets"][1] + base)
        for k, v in meta.items() if k != "__metadata__"
    }


def _read_safetensors_meta(path: str) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    """{tensor_name: (abs_start, abs_end, shape)} -- same math as
    ``_read_safetensors_offsets`` plus the declared shape (exl3 needs it for the
    per-layer K, without loading any tensor)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
    base = 8 + hlen
    out = {}
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        out[k] = (
            v["data_offsets"][0] + base,
            v["data_offsets"][1] + base,
            tuple(v.get("shape", ())),
        )
    return out


# routed EXL3 expert tensors: trellis / suh / svh under experts.<e>.<proj>_proj.*
# (the anchor ``layers.N.mlp.experts.E.`` keeps out the MTP head and shared experts)
_EXL3_KEY_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.(?P<kind>trellis|suh|svh)$"
)



class Nvfp4DiskIndex:
    """(bank, layer, expert) -> per-segment (shard_idx, offset, nbytes) locations.

    Built from the original checkpoint: the HF index json (name -> shard) plus
    each referenced shard's safetensors header (name -> byte range). Expert
    tensors are per-expert and contiguous, so a row is exactly one byte range
    per segment.
    """

    def __init__(self, model_dir: str, config, spec) -> None:
        from freetoken.models.nvfp4_banks import _canon_kind, _num_moe_layers
        from freetoken.utils.hf import download_hf_weight

        model_dir = download_hf_weight(model_dir)  # hub id -> local cache dir; no-op if local
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        num_layers = _num_moe_layers(config)
        # (bank_layer, expert, proj, kind) -> (tensor_name, shard). Kind names are
        # canonicalized (compressed-tensors' weight_packed -> weight, ...) so the
        # bank segment tables below match however the checkpoint spells them.
        loc: dict[tuple[int, int, str, str], tuple[str, str]] = {}
        for name, shard in weight_map.items():
            m = spec.key_pattern.match(name)
            if m is None:
                continue
            bank_layer = spec.layer_to_bank(int(m.group("layer")), config)
            if bank_layer is None:
                continue
            loc[(bank_layer, int(m.group("expert")), m.group("proj"),
                 _canon_kind(spec, m.group("kind")))] = (name, shard)
        # compressed-tensors stores the QUANT-side global scale; the banks keep
        # its reciprocal (matching nvfp4_banks._ingest_global).
        self.global_reciprocal = bool(getattr(spec, "global_reciprocal", False))

        shards = sorted(set(shard for _, shard in loc.values()))
        self.shard_paths = [os.path.join(model_dir, s) for s in shards]
        offsets = {s: _read_safetensors_offsets(os.path.join(model_dir, s)) for s in shards}
        shard_idx = {s: i for i, s in enumerate(shards)}

        E = config.num_experts
        seg_size = struct.calcsize("<iqq")  # (shard_idx, offset, nbytes)
        self.entries: list[list[bytes]] = []  # [bank][layer] -> packed segments per expert
        for bank_idx in range(len(_NVP4_BANK_SEGS)):
            per_layer = []
            for layer in range(num_layers):
                rows = bytearray()
                for e in range(E):
                    for proj, kind, _, _ in _NVP4_BANK_SEGS[bank_idx]:
                        key = (layer, e, proj, kind)
                        entry = loc.get(key)
                        if entry is None:
                            raise KeyError(
                                f"disk tier: no {proj}.{kind} tensor for layer {layer} expert {e} "
                                f"(bank {bank_idx}) in {index_path}"
                            )
                        name, shard = entry
                        start, end = offsets[shard][name]
                        rows += struct.pack("<iqq", shard_idx[shard], start, end - start)
                per_layer.append(bytes(rows))
            self.entries.append(per_layer)
        self._seg_size = seg_size

    def row_segments(self, bank_idx: int, layer: int, expert: int) -> list[tuple[int, int, int]]:
        """[(shard_idx, offset, nbytes)] for one expert row, in segment order."""
        base = expert * self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])
        raw = self.entries[bank_idx][layer][base:base + self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])]
        return [
            struct.unpack_from("<iqq", raw, i * self._seg_size)
            for i in range(len(_NVP4_BANK_SEGS[bank_idx]))
        ]


class _Exl3Seg:
    __slots__ = ("shard_idx", "offset", "nbytes", "dst_off", "field")

    def __init__(self, shard_idx: int, offset: int, nbytes: int, dst_off: int, field: int) -> None:
        self.shard_idx = shard_idx
        self.offset = offset
        self.nbytes = nbytes
        self.dst_off = dst_off
        # full field width in the padded row: bytes [nbytes, field) beyond the
        # checkpoint tensor must read as ZERO on the bank row (bytes [field,)
        # are the next field and belong to another segment)
        self.field = field


# exl3 row layout in terms of (H, I, tb_max): field names -> byte offsets.
# gate_up: gate.trellis | up.trellis | gate.suh | gate.svh | up.suh | up.svh
# down:    down.trellis | down.suh | down.svh
def _exl3_field_offsets() -> list[dict[str, str]]:
    return [
        {
            "gate_trellis": "0",
            "up_trellis": "tb_max",
            "gate_suh": "2*tb_max",
            "gate_svh": "2*tb_max+2*H",
            "up_suh": "2*tb_max+2*H+2*I",
            "up_svh": "2*tb_max+4*H+2*I",
            "tb_max": "tb_max",
        },
        {
            "down_trellis": "0",
            "down_suh": "tb_max",
            "down_svh": "tb_max+2*I",
            "tb_max": "tb_max",
        },
    ]


class Exl3DiskIndex:
    """(bank, layer, expert) -> per-segment (shard, offset, nbytes, dst_off, field).

    EXL3 rows are spliced from per-expert checkpoint tensors. The trellis field
    is read at this layer's K (a mixed-rate recipe reads ``tb < tb_max`` bytes)
    and padded with ZEROS to ``tb_max`` -- the row layout of every layer is
    built for the checkpoint's MAX K (see _setup_exl3_banks). The fp16 suh/svh
    sign scales sit at fixed byte offsets and copy 1:1. Per-layer K comes from
    the shard safetensors headers, so building the index reads metadata only.
    """

    # segments per bank: (proj, kind, dst_off_terms, pads-to-field). Merely
    # positive terms; off = sum of terms; "tbw" marks the trellis width base.
    _SEGS = (
        (
            ("gate_proj", "trellis", (), True),
            ("up_proj", "trellis", ("tb_max",), True),
            ("gate_proj", "suh", ("tb_max", "tb_max"), False),
            ("gate_proj", "svh", ("tb_max", "tb_max"), False),
            ("up_proj", "suh", ("tb_max", "tb_max"), False),
            ("up_proj", "svh", ("tb_max", "tb_max"), False),
        ),
        (
            ("down_proj", "trellis", (), True),
            ("down_proj", "suh", ("tb_max",), False),
            ("down_proj", "svh", ("tb_max",), False),
        ),
    )
    _ROW_NAMES = ("gate_up", "down")

    def __init__(self, model_dir: str, config) -> None:
        from freetoken.models.nvfp4_banks import _num_moe_layers
        from freetoken.models.qwen3_5_moe.weight import _moe_dims
        from freetoken.utils.hf import download_hf_weight

        model_dir = download_hf_weight(model_dir)
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        L, E, H, I, dense = _moe_dims(config)
        num_layers = _num_moe_layers(config)
        loc: dict[tuple[int, int, str, str], tuple[str, str]] = {}
        for name, shard in weight_map.items():
            m = _EXL3_KEY_RE.match(name)
            if m is None:
                continue
            layer = int(m.group("layer")) - dense
            if not (0 <= layer < num_layers):
                continue
            loc[(layer, int(m.group("expert")), m.group("proj") + "_proj",
                 m.group("kind"))] = (name, shard)

        shards = sorted(set(shard for _, shard in loc.values()))
        self.shard_paths = [os.path.join(model_dir, s) for s in shards]
        meta = {sh: _read_safetensors_meta(os.path.join(model_dir, sh)) for sh in shards}
        shard_idx = {sh: i for i, sh in enumerate(shards)}

        # pass 1: per-layer K (trellis shape of expert 0, stored int16 windows
        # (H//16, I//16, K*16) -> trellis bytes tb = shape0*shape1*shape2*2)
        tbs: list[int] = []
        for layer in range(num_layers):
            gkey = (layer, 0, "gate_proj", "trellis")
            if gkey not in loc:
                raise KeyError(
                    f"disk tier: no exl3 gate_proj.trellis for layer {layer} in {index_path}")
            gname, gshard = loc[gkey]
            _, _, ts = meta[gshard][gname]
            if (ts[0] * 16, ts[1] * 16) != (H, I):
                raise ValueError(
                    f"disk tier: exl3 layer {layer} trellis geometry {ts[0] * 16}x{ts[1] * 16}"
                    f" != config {H}x{I}")
            tb = ts[0] * ts[1] * ts[2] * 2
            if tb <= 0 or tb > H * I:
                raise ValueError(f"disk tier: exl3 layer {layer} trellis {tb} B bad")
            tbs.append(tb)
        k_per_layer = tuple(tb * 8 // (H * I) for tb in tbs)
        self.k_per_layer = k_per_layer
        tb_max = max(tbs)
        row_gu = 2 * tb_max + 4 * H + 4 * I
        row_dn = tb_max + 2 * H + 2 * I
        row_bytes = (row_gu, row_dn)
        # (proj, kind) -> dst byte offset within the padded row
        dst_gu = {
            ("gate_proj", "trellis"): 0,
            ("up_proj", "trellis"): tb_max,
            ("gate_proj", "suh"): 2 * tb_max,
            ("gate_proj", "svh"): 2 * tb_max + 2 * H,
            ("up_proj", "suh"): 2 * tb_max + 2 * H + 2 * I,
            ("up_proj", "svh"): 2 * tb_max + 4 * H + 2 * I,
        }
        dst_dn = {
            ("down_proj", "trellis"): 0,
            ("down_proj", "suh"): tb_max,
            ("down_proj", "svh"): tb_max + 2 * I,
        }
        dsts = (dst_gu, dst_dn)
        seg_counts = tuple(len(seg) for seg in self._SEGS)

        def _field_width(bank: int, kind: str) -> int:
            if kind == "trellis":
                return tb_max
            if bank == 0:  # gate/up: input axis H, output axis I
                return 2 * H if kind == "suh" else 2 * I
            # down: input axis I, output axis H
            return 2 * I if kind == "suh" else 2 * H

        # pass 2: pack segments [(shard_idx, offset, nbytes, dst_off, field)] per
        # expert row in bank order
        self._seg_size = struct.calcsize("<iqqQQ")
        self.row_seg_counts = seg_counts
        self.entries: list[list[bytes]] = []
        for bank, bank_segs in enumerate(self._SEGS):
            n_seg = len(bank_segs)
            src = row_gu if bank == 0 else row_dn
            per_layer: list[bytes] = []
            for layer in range(num_layers):
                tb = tbs[layer]
                rows = bytearray()
                for e in range(E):
                    for proj, kind, _, pads in bank_segs:
                        key = (layer, e, proj, kind)
                        entry = loc.get(key)
                        if entry is None:
                            raise KeyError(
                                f"disk tier: no exl3 {proj}.{kind} tensor for layer {layer}"
                                f" expert {e} in {index_path}")
                        tname, tshard = entry
                        tstart, tend, _ts = meta[tshard][tname]
                        nbytes = tend - tstart
                        expected = _field_width(bank, kind)
                        if pads and nbytes != tb:
                            raise ValueError(
                                f"disk tier: exl3 L{layer} {proj} trellis {nbytes} B != {tb} B (K of e0)")
                        if not pads and nbytes != expected:
                            raise ValueError(
                                f"disk tier: exl3 L{layer} {proj}.{kind} {nbytes} B != {expected} B")
                        dst_off = dsts[bank][(proj, kind)]
                        if dst_off + expected > src:
                            raise ValueError(
                                f"disk tier: exl3 L{layer} {proj}.{kind} field overruns its row")
                        rows += struct.pack("<iqqQQ", shard_idx[tshard], tstart, nbytes,
                                            dst_off, expected)
                per_layer.append(bytes(rows))
            self.entries.append(per_layer)
        self.row_bytes = {"gate_up": row_gu, "down": row_dn}
        self.tb_max = tb_max

    def row_segments(self, bank_idx: int, layer: int, expert: int
                     ) -> list[tuple[int, int, int, int, int]]:
        """[(shard_idx, offset, nbytes, dst_off, field)] for one expert row.

        ``field`` is the full padded-field width: bytes [nbytes, field) beyond
        the checkpoint tensor must read as ZERO (mixed-rate trellis); scales
        carry field == nbytes."""
        n_seg = self.row_seg_counts[bank_idx]
        base = expert * self._seg_size * n_seg
        raw = self.entries[bank_idx][layer][base:base + self._seg_size * n_seg]
        return [
            struct.unpack_from("<iqqQQ", raw, i * self._seg_size)
            for i in range(n_seg)
        ]
class DiskTier:
    """Runtime fetcher: disk-resident slot-cache misses -> staging -> GPU slot."""

    def __init__(self, index: Nvfp4DiskIndex, cache, ram_experts: int, workers: int = 8,
                 perm: "torch.Tensor | None" = None) -> None:
        self._index = index
        self._ram = ram_experts
        self._exl3 = isinstance(index, Exl3DiskIndex)
        # Learned hot-set row order: ``perm[layer, expert] -> row``. The cache
        # hands us ROW ids; the checkpoint index speaks EXPERT ids, so keep the
        # inverse for the translation. None == identity ordering.
        self._perm = perm
        self._perm_inv = None
        if perm is not None:
            self._perm_inv = torch.empty_like(perm)
            self._perm_inv.scatter_(
                1, perm, torch.arange(perm.shape[1]).expand(perm.shape[0], perm.shape[1]))
        self._banks = list(cache.banks)  # [(per_layer_host, gpu_cache)] in schema order
        self._row_bytes = [
            b[0][0][0].numel() * b[0][0][0].element_size() for b in self._banks
        ]  # full expert-row bytes per bank (staging must hold the biggest one)
        # Per-bank destination row slices (gate|up split at the row midpoint); exl3
        # rows carry their own byte destinations on each segment instead.
        self._dst_slices: list[list[tuple[int, int]]] = []
        for bank_idx, (host_layer, _gpu) in enumerate(self._banks):
            row = host_layer[0][0]
            if self._exl3:
                self._dst_slices.append([(0, row.numel())])
            elif len(_NVP4_BANK_SEGS[bank_idx]) == 2:
                mid = row.shape[0] // 2
                self._dst_slices.append([(0, mid), (mid, row.shape[0])])
            else:
                self._dst_slices.append([(0, row.shape[0])])
        if os.environ.get("FT_DISK_TIER_VERIFY"):
            # TP=2 debug: prove per-rank whether the host bank rows are full or
            # TP-sharded, and that the disk index's full-row segments match them.
            from freetoken.distributed import try_get_tp_info
            tp = try_get_tp_info()
            host_shapes = [tuple(b[0][0][0].shape) for b in self._banks]
            disk_bytes = [
                sum(nb for _, _, nb in self._index.row_segments(bi, 0, 0))
                for bi in range(len(self._banks))
            ]
            print(f"[disk-tier-init] tp_rank={getattr(tp, 'rank', '?')}/{getattr(tp, 'size', '?')} "
                  f"ram={ram_experts} host_row_shapes={host_shapes} "
                  f"host_row_bytes={self._row_bytes} disk_row_bytes={disk_bytes}", flush=True)
        max_row = max(self._row_bytes)
        self._staging_size = ((max_row + _ALIGN - 1) // _ALIGN + 2) * _ALIGN
        self._staging = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="disk-tier")
        self._fd_lock = threading.Lock()
        self._fds: dict[int, tuple[int, bool]] = {}
        self._fetches = 0
        self._fetch_bytes = 0
        self._decode_verify_steps = 0
        self._map_verify_steps = 0
        self._cache = cache
        # Temporal prefetch (FT_DISK_PREFETCH_ROWS = total pinned rows to keep; 0 = off).
        self._pf_budget = int(os.environ.get("FT_DISK_PREFETCH_ROWS", "0"))
        self._pf: dict[tuple[int, int], list[HostBank]] = {}
        self._pf_order: list[tuple[int, int]] = []
        self._pf_inflight: dict[tuple[int, int], object] = {}
        self._pf_lock = threading.Lock()
        self._pf_hits = 0
        self._pf_fetches = 0
        self._last_rows: dict[int, tuple[int, ...]] = {}
        self._pf_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="disk-tier-pf")

    # ------------------------------------------------------------------ fds
    def _fd(self, shard_idx: int) -> tuple[int, bool]:
        """(fd, o_direct) for a shard; O_DIRECT falls back to plain preadv where the
        filesystem refuses it (tmpfs/overlayfs -- tests)."""
        ent = self._fds.get(shard_idx)
        if ent is None:
            with self._fd_lock:
                ent = self._fds.get(shard_idx)
                if ent is None:
                    path = self._index.shard_paths[shard_idx]
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                        direct = True
                    except OSError:
                        fd = os.open(path, os.O_RDONLY)
                        direct = False
                    ent = (fd, direct)
                    self._fds[shard_idx] = ent
        return ent

    # --------------------------------------------------------------- staging
    # Staging ring depth per worker thread. A buffer must not be overwritten by the
    # next preadv until the async H2D copy that read it has finished (pinned-memory
    # reuse race -- the copy is DMA, still reading host bytes after copy_ returns).
    # The depth only needs to cover one copy's DMA time in host-side preadv time;
    # the per-slot CUDA event below makes any shallower lap correct, just slower.
    _STAGING_RING = 8

    def _staging_ring(self) -> list:
        ring = getattr(self._staging, "ring", None)
        if ring is None:
            # Fresh worker threads default to CUDA device 0, but the rank may
            # live on another device (TP>1). The H2D copies land on the
            # destination tensor's device stream, while ev.record() below uses
            # the thread's CURRENT stream -- without this, the ring's reuse
            # guard waits on an idle stream and a preadv can overwrite the
            # buffer mid-DMA (corrupted slot rows on TP=2 rank 1).
            dev = self._banks[0][1].device
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            ring = []
            for _ in range(self._STAGING_RING):
                buf = HostBank((self._staging_size,), torch.uint8)
                buf.pin()  # pin once per worker thread
                ev = torch.cuda.Event() if torch.cuda.is_available() else None
                if ev is not None:
                    ev.record()  # start "complete"; re-recorded after each copy
                ring.append([buf, ev])
            self._staging.ring = ring
        return ring

    # ---------------------------------------------------------------- fetch
    def _fetch_expert(self, layer: int, expert: int, slot: int) -> None:
        # The server runs under inference_mode; the fetch pool threads do not,
        # so the H2D writes into the (inference) slot cache need their own scope.
        with torch.inference_mode():
            self._fetch_expert_inner(layer, expert, slot)

    def _pack_row(self, layer: int, expert: int, rows: list[torch.Tensor]) -> None:
        """Read one expert row from the checkpoint into ``rows`` (one per bank),
        in the packed byte layout the slot cache expects. Used both for the
        demand fetch (rows = GPU slots) and for prefetch (rows = host buffers)."""
        ring = self._staging_ring()
        ri = getattr(self._staging, "ri", 0)
        for bank_idx, row in enumerate(rows):
            if self._exl3:
                # exl3 segments carry their own byte destinations; trellis
                # segments pad their field with zeros (mixed-rate K < k_max).
                segs = self._index.row_segments(bank_idx, layer, expert)
                for (shard_idx, off, nbytes, dst_off, field) in segs:
                    staging, ev = ring[ri]
                    if ev is not None:
                        ev.synchronize()
                    ri = (ri + 1) % len(ring)
                    fd, direct = self._fd(shard_idx)
                    if direct:
                        a0 = off & ~(_ALIGN - 1)
                        slen = (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
                    else:
                        a0, slen = off, nbytes
                    mv = (ctypes.c_char * slen).from_address(staging.addr)
                    try:
                        os.preadv(fd, [mv], a0)
                    except OSError:
                        raise _preadv_error(self, staging, shard_idx, off, a0, slen, direct)
                    row_off = off - a0
                    dst = row.view(torch.uint8)
                    src = staging.tensor[row_off:row_off + nbytes]
                    dst[dst_off:dst_off + nbytes].copy_(
                        src.view(dst.dtype), non_blocking=True)
                    if field > nbytes:
                        # mixed-rate trellis: zero the tail of the padded field
                        dst[dst_off + nbytes:dst_off + field].zero_()
                    if ev is not None:
                        ev.record()
                continue
            segs = self._index.row_segments(bank_idx, layer, expert)
            for (d0, d1), (shard_idx, off, nbytes) in zip(self._dst_slices[bank_idx], segs):
                staging, ev = ring[ri]
                if ev is not None:
                    # This buffer's last async H2D copy must be done before the
                    # preadv below overwrites it (pinned-memory reuse race).
                    ev.synchronize()
                ri = (ri + 1) % len(ring)
                fd, direct = self._fd(shard_idx)
                if direct:
                    a0 = off & ~(_ALIGN - 1)
                    slen = (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
                else:
                    a0, slen = off, nbytes
                mv = (ctypes.c_char * slen).from_address(staging.addr)
                try:
                    os.preadv(fd, [mv], a0)
                except OSError:
                    raise _preadv_error(self, staging, shard_idx, off, a0, slen, direct)
                row_off = off - a0
                if bank_idx in (2, 5):
                    # Global-scale banks: the checkpoint stores a per-expert fp32
                    # SCALAR (weight_scale_2); the bank row is that value as fp16
                    # broadcast across the row -- convert + fill, no byte copy.
                    # compressed-tensors stores the quant-side scale, so invert
                    # it first (see nvfp4_banks._ingest_global).
                    val = staging.tensor[row_off:row_off + 4].view(torch.float32)[0]
                    if getattr(self._index, "global_reciprocal", False):
                        val = 1.0 / val
                    row[d0:d1].fill_(val.to(torch.float16))
                    continue
                src = staging.tensor[row_off:row_off + nbytes]
                dst = row[d0:d1]
                dst.copy_(src.view(dst.dtype).view(dst.shape), non_blocking=True)
                if ev is not None:
                    # Arm: the next reuse of this buffer waits for this copy.
                    ev.record()
        self._staging.ri = ri

    def _fetch_expert_inner(self, layer: int, expert: int, slot: int) -> None:
        rows = [gpu_cache[slot] for (_host, gpu_cache) in self._banks]
        packed = self._take_prefetched(layer, expert)
        if packed is not None:
            # Prefetched earlier (predicted from the previous token): copy the packed
            # host row into the slot instead of waiting on the disk.
            for buf, row in zip(packed, rows):
                row.reshape(-1).view(torch.uint8).copy_(buf.tensor, non_blocking=True)
        else:
            self._pack_row(layer, expert, rows)
        self._fetches += 1
        self._fetch_bytes += sum(self._row_bytes)

    # ------------------------------------------------------------ prefetch
    # Temporal lookahead (Colibri's PILOT, simplified): the disk rows a layer
    # needed for the last token predict the next token's (routing persists across
    # steps). Prefetch them into pinned host buffers in the background so the
    # demand fetch is an H2D copy instead of a disk read. Prediction errors only
    # waste I/O -- the demand path always falls back to reading the checkpoint.
    def _new_pf_buffers(self) -> list[HostBank]:
        bufs = []
        for nbytes in self._row_bytes:
            buf = HostBank((nbytes,), torch.uint8)
            try:
                buf.pin()
            except Exception:
                pass  # pageable is fine (slower H2D) on hosts without pinning
            bufs.append(buf)
        return bufs

    def _take_prefetched(self, layer: int, row: int) -> list[HostBank] | None:
        if self._pf_budget <= 0:
            return None
        key = (layer, row)
        while True:
            with self._pf_lock:
                bufs = self._pf.pop(key, None)
                if bufs is not None:
                    try:
                        self._pf_order.remove(key)
                    except ValueError:
                        pass
                    self._pf_hits += 1
                    return bufs
                fut = self._pf_inflight.get(key)
            if fut is None:
                return None
            fut.result()  # prefetch in flight: wait, then re-check the store

    def _prefetch_row(self, layer: int, row: int) -> None:
        try:
            with torch.inference_mode():
                bufs = self._new_pf_buffers()
                self._pack_row(layer, row, [b.tensor for b in bufs])
        except Exception:
            with self._pf_lock:
                self._pf_inflight.pop((layer, row), None)
            return
        key = (layer, row)
        with self._pf_lock:
            if self._pf_budget <= 0:
                self._pf_inflight.pop(key, None)
                return
            self._pf[key] = bufs
            self._pf_order.append(key)
            # evict oldest beyond the budget (the just-stored key is youngest)
            while len(self._pf_order) > self._pf_budget:
                old = self._pf_order.pop(0)
                self._pf.pop(old, None)
            self._pf_inflight.pop(key, None)
        self._pf_fetches += 1

    def _schedule_prefetch(self, layer: int, rows) -> None:
        if self._pf_budget <= 0:
            return
        # demand fetches are keyed by (layer, EXPERT); rows and experts are a
        # bijection under the learned perm, so translate once here
        experts = tuple(self._expert_of(int(r), layer) for r in rows)
        self._last_rows[layer] = experts
        for expert in experts:
            key = (layer, expert)
            with self._pf_lock:
                if key in self._pf or key in self._pf_inflight:
                    continue
                self._pf_inflight[key] = self._pf_pool.submit(
                    self._prefetch_row, layer, expert)

    def _sync_fetches(self) -> None:
        """Wait for the pool threads' async H2D copies to land.

        The copies are enqueued on the pool threads' default stream; f.result() only
        waits for them to be ENQUEUED. The GEMM's stream is not ordered with that
        stream, so sync the default stream before the GEMM reads the slots."""
        # Key this on where the BANKS live, not on whether the machine has a GPU: the
        # disk-tier unit tests build CPU banks on a CUDA box, and default_stream() rejects
        # a CPU device outright. There is nothing to order for a CPU->CPU copy anyway.
        device = self._banks[0][1].device
        if device.type != "cuda":
            return  # CPU banks: the copies are synchronous CPU->CPU
        torch.cuda.default_stream(device).synchronize()

    def _verify_slot(self, cache, layer: int, expert: int, slot: int | None = None,
                     phase: str = "prefill") -> None:
        """One-shot debug: read back an expert's slot rows and compare against the
        checkpoint bytes (ground truth). Gated on FT_DISK_TIER_VERIFY."""
        import torch
        if slot is None:
            slot = expert  # identity mapping (prefill)
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            slot_row = gpu_cache[slot].contiguous()
            flat = slot_row.view(torch.uint8).reshape(-1)
            ref = self._ref_row(bank_idx, layer, expert, flat.numel(),
                                slot_row.element_size(),
                                slot_row.numel() // slot_row.shape[0] if slot_row.dim() > 1 else 1)
            try:
                ref = ref.to(flat.device)
                match = bool(torch.equal(flat, ref))
                if match:
                    print(f"[verify] {phase} L{layer} bank={bank_idx} expert={expert} "
                          f"slot={slot} match=True", flush=True)
                    continue
                diff = (flat != ref)
                nz = torch.nonzero(diff).flatten()
                print(f"[verify] {phase} L{layer} bank={bank_idx} expert={expert} "
                      f"slot={slot} match=False n_diff={int(diff.sum())}/{flat.numel()} "
                      f"first_off={int(nz[0])} last_off={int(nz[-1])} "
                      f"slot_norm={slot_row.float().norm().item():.4f} "
                      f"ref_norm={ref.view(slot_row.dtype).view(slot_row.shape).float().norm().item():.4f} "
                      f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}", flush=True)
                self._identify_overwriter(layer, bank_idx, expert, flat)
            except Exception as exc:  # never crash the server in debug
                print(f"[verify] bank={bank_idx} expert={expert} ERROR {exc!r}", flush=True)

    def _ref_row(self, bank_idx: int, layer: int, expert: int, row_bytes: int,
                 row_el: int, row_leading: int) -> torch.Tensor:
        """Reference row bytes for (bank, layer, expert) straight from the checkpoint."""
        ref = torch.zeros(row_bytes, dtype=torch.uint8)
        segs = self._index.row_segments(bank_idx, layer, expert)
        if self._exl3:
            for (shard_idx, off, nbytes, dst_off, _field) in segs:
                fd, direct = self._fd(shard_idx)
                a0 = off if not direct else (off & ~(_ALIGN - 1))
                slen = nbytes if not direct else (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
                buf = os.pread(fd, slen, a0)
                seg = buf[off - a0:off - a0 + nbytes]
                ref[dst_off:dst_off + nbytes] = torch.frombuffer(seg, dtype=torch.uint8)
            return ref
        for (d0, d1), (shard_idx, off, nbytes) in zip(self._dst_slices[bank_idx], segs):
            fd, direct = self._fd(shard_idx)
            a0 = off if not direct else (off & ~(_ALIGN - 1))
            slen = nbytes if not direct else (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
            buf = os.pread(fd, slen, a0)
            row_off = off - a0
            seg = buf[row_off:row_off + nbytes]
            if bank_idx in (2, 5):
                import struct as _st
                import numpy as _np
                val = _st.unpack("<f", seg[:4])[0]
                if getattr(self._index, "global_reciprocal", False):
                    val = 1.0 / val
                f16 = _np.float16(val).tobytes()
                for r in range(d0, d1):
                    ref[r * row_el:(r + 1) * row_el] = torch.frombuffer(f16, dtype=torch.uint8)
            else:
                dst_off = d0 * row_leading * row_el
                ref[dst_off:dst_off + len(seg)] = torch.frombuffer(seg, dtype=torch.uint8)
        return ref

    def _identify_overwriter(self, layer: int, bank_idx: int, expert: int,
                             flat: torch.Tensor) -> None:
        """Debug: on a verify mismatch, find whose row the slot actually holds.

        Compares the slot's first 64 bytes against (a) every other expert of the
        same layer and (b) the same expert in every other layer, straight from the
        checkpoint. A hit names the overwriter (e.g. a staging-buffer reuse race
        landing a neighbour's preadv); no hit means a partial mix. Only runs on
        mismatch, so the ~300 extra preads are free otherwise."""
        head = flat[:64].cpu()
        host_row = self._banks[bank_idx][0][0][0]  # expert-0 row (all rows share its shape)
        row_el = host_row.element_size()
        row_leading = host_row.numel() // host_row.shape[0] if host_row.dim() > 1 else 1
        num_experts = self._cache.num_experts
        num_layers = len(self._banks[bank_idx][0])
        hits = []
        for e in range(num_experts):
            if e == expert:
                continue
            ref = self._ref_row(bank_idx, layer, e, flat.numel(), row_el, row_leading)
            if bool(torch.equal(ref[:64], head)):
                hits.append(f"L{layer}_e{e}")
        for L in range(num_layers):
            if L == layer:
                continue
            ref = self._ref_row(bank_idx, L, expert, flat.numel(), row_el, row_leading)
            if bool(torch.equal(ref[:64], head)):
                hits.append(f"L{L}_e{expert}")
        print(f"[overwriter] L{layer} B{bank_idx} e{expert} head64 matches: "
              f"{hits if hits else 'NONE (partial mix?)'}", flush=True)

    def verify_ram(self, cache, layer: int) -> None:
        """One-shot debug: after the PCIe copy, check a RAM-resident expert's slot rows
        against the checkpoint reference. Gated on FT_DISK_TIER_VERIFY."""
        expert = min(10, self._ram - 1)  # a RAM-resident expert
        print(f"[verify-ram] layer={layer} expert={expert} (RAM prefix)", flush=True)
        self._verify_slot(cache, layer, expert)
        # Check ALL RAM experts: slot vs host row (host correctness established
        # separately). Count mismatches; identify the source of the first one.
        n_bad = 0
        identified = False
        for e in range(self._ram):
            for bank_idx, (host_layer, gpu_cache) in enumerate(self._banks):
                slot_row = gpu_cache[e].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                hflat = host_layer[layer][e].contiguous().view(torch.uint8).reshape(-1)
                if flat.numel() != hflat.numel() or not bool(torch.equal(flat.cpu(), hflat)):
                    n_bad += 1
                    if n_bad <= 12:
                        print(f"[verify-ram] MISMATCH e={e} bank={bank_idx} "
                              f"slot_head={flat[:8].tolist()} host_head={hflat[:8].tolist()}",
                              flush=True)
                    if not identified:
                        identified = True
                        self._identify_source(layer, bank_idx, flat)
        print(f"[verify-ram] layer={layer} mismatches={n_bad}/{self._ram * len(self._banks)}",
              flush=True)

    def _identify_source(self, layer: int, bank_idx: int, flat: torch.Tensor) -> None:
        """Debug: find where a corrupted slot's bytes came from (GPU slot, host row,
        or checkpoint row)."""
        flat_cpu = flat.cpu()
        head = flat_cpu[:16]
        found = []
        _host_layer, gpu_cache = self._banks[bank_idx]
        gpu_flat = gpu_cache.view(torch.uint8).reshape(gpu_cache.shape[0], -1)
        if gpu_flat.shape[1] == flat_cpu.numel():
            cand = torch.nonzero(
                (gpu_flat[:, :16].cpu() == head.unsqueeze(0)).all(dim=1)).flatten().tolist()
            for s in cand[:16]:
                if bool(torch.equal(gpu_flat[s].cpu(), flat_cpu)):
                    found.append(f"gpu_slot={s}(L{layer},B{bank_idx})")
        for e in range(self._ram):
            hflat = _host_layer[layer][e].contiguous().view(torch.uint8).reshape(-1)
            if hflat.numel() == flat_cpu.numel() and bool(torch.equal(hflat, flat_cpu)):
                found.append(f"host_row_e{e}(L{layer},B{bank_idx})")
        import itertools
        num_layers = len(self._banks[bank_idx][0])
        targets = sorted(set(itertools.product([layer], range(256)))
                         | set(itertools.product(range(num_layers), [10])))
        for (L, e) in targets:
            row_el, row_leading = None, None
            hrow = _host_layer[layer][0]
            row_el = hrow.element_size()
            row_leading = hrow.numel() // hrow.shape[0] if hrow.dim() > 1 else 1
            try:
                ref = self._ref_row(bank_idx, L, e, flat_cpu.numel(), row_el, row_leading)
            except Exception:
                continue
            if bool(torch.equal(ref, flat_cpu)):
                found.append(f"checkpoint_L{L}_e{e}")
        print(f"[identify] L{layer} B{bank_idx} n={flat_cpu.numel()} "
              f"source={found if found else 'UNKNOWN'}", flush=True)

    def verify_decode_mapping(self, cache, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Debug: after the LRU rewrite + fetch/copy, check that every slot the GEMM
        will read actually holds the expert the bookkeeping says it holds. Gated on
        FT_DISK_TIER_VERIFY; first 4 decode steps only."""
        if self._map_verify_steps >= 4:
            return
        self._map_verify_steps += 1
        slots = torch.unique(topk_ids.reshape(-1))
        nbad = 0
        for s in slots.tolist():
            s = int(s)
            flat_id = int(cache.id_of_slot[s].item())
            if flat_id < 0:
                print(f"[verify-map] step={self._map_verify_steps} slot={s} id_of_slot=-1",
                      flush=True)
                nbad += 1
                continue
            row = flat_id % cache.num_experts
            expert = self._expert_of(row, layer_id)
            for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
                slot_row = gpu_cache[s].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                ref = self._ref_row(bank_idx, layer_id, expert, flat.numel(),
                                    slot_row.element_size(),
                                    slot_row.numel() // slot_row.shape[0]
                                    if slot_row.dim() > 1 else 1)
                if not bool(torch.equal(flat.cpu(), ref)):
                    nbad += 1
                    if nbad <= 8:
                        print(f"[verify-map] step={self._map_verify_steps} slot={s} "
                              f"expert={expert} bank={bank_idx} MISMATCH "
                              f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}",
                              flush=True)
        print(f"[verify-map] step={self._map_verify_steps} slots={slots.numel()} bad={nbad}",
              flush=True)

    def _expert_of(self, row: int, layer: int) -> int:
        """Checkpoint expert id for a bank row (identity without a learned perm)."""
        if self._perm_inv is None:
            return row
        return int(self._perm_inv[layer, row])

    def materialize_layer(self, cache, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Disk-tier prefill: materialize the RAM-resident prefix into identity slots
        (the normal kernel restricted to K experts; the following ``copy_missing``
        streams it over PCIe), then fetch the routed disk-resident experts into
        THEIR identity slots. The identity mapping (position == expert id) is
        preserved, so the prefill GEMM is unchanged."""
        from freetoken.moe.offload_kernels import _materialize_layer_gpu

        # Prefill identity mapping owns ALL of slots [0, E) for this layer, but
        # the kernel only scans slots < materialize_count, so the disk slots
        # [ram, E) that still hold a previous layer's experts (previous prefill
        # layer or decode LRU) would keep their slot_for_id entries -- phantom
        # decode hits that read another layer's weights. Clear them first
        # (device-side, no sync).
        seg = cache.id_of_slot[self._ram:cache.num_experts]
        valid = seg >= 0
        cache.slot_for_id.view(-1)[seg[valid].long()] = -1
        seg[valid] = -1
        cache.usage[self._ram:cache.num_experts][valid] = 0

        _materialize_layer_gpu(cache, layer_id, materialize_count=self._ram)
        routed = expert_ids.reshape(-1)
        disk = torch.unique(routed[routed >= self._ram])
        if os.environ.get("FT_DISK_TIER_DEBUG") and layer_id < 3:
            print(f"[disk-tier dbg] layer={layer_id} routed={routed.numel()} "
                  f"unique_disk={disk.numel()} disk={disk.tolist()[:12]}", flush=True)
        if disk.numel() == 0:
            return
        # cache.step was already incremented by the kernel; assign the 0-d tensor
        # device-side (same dtype/device as usage) instead of .item()-ing it, which
        # would sync the stream once per layer on the prefill/decode path.
        futures = [
            self._pool.submit(self._fetch_expert, layer_id,
                              self._expert_of(int(row), layer_id), int(row))
            for row in disk.tolist()
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        if os.environ.get("FT_DISK_TIER_VERIFY") and layer_id in (0, 20) and disk.numel() > 0:
            limit = disk.numel() if layer_id == 0 else 6  # layer 0: ALL experts (race hunt)
            for row in disk.tolist()[:limit]:
                self._verify_slot(cache, layer_id, self._expert_of(int(row), layer_id),
                                  slot=int(row), phase="prefill")
        # Same bookkeeping the materialize kernel writes, per fetched expert.
        flat = layer_id * cache.num_experts + disk
        cache.slot_for_id[layer_id, disk] = disk
        cache.id_of_slot[disk] = flat
        cache.usage[disk] = cache.step

    def fetch_pending(self, cache, layer_id: int) -> None:
        """Fetch this layer's disk-resident misses into their slots; shrink the miss
        list to the RAM-resident remainder for the existing PCIe copy path."""
        n = int(cache.num_indices.item())
        if n == 0:
            return
        src = cache.src_indices[:n].cpu()
        slots = cache.evict_slots[:n].cpu()
        disk = [i for i in range(n) if int(src[i]) >= self._ram]
        if os.environ.get("FT_DISK_TIER_VERIFY") and layer_id == 0:
            print(f"[fetch-pend] layer=0 n={n} ndisk={len(disk)} "
                  f"src_head={src[:4].tolist()}", flush=True)
        if not disk:
            return
        futures = [
            self._pool.submit(self._fetch_expert, layer_id,
                              self._expert_of(int(src[i]), layer_id), int(slots[i]))
            for i in disk
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        # Predict the next token: the rows this layer missed are likely needed again.
        self._schedule_prefetch(layer_id, [int(src[i]) for i in disk])
        if (os.environ.get("FT_DISK_TIER_VERIFY") and layer_id == 0
                and self._decode_verify_steps < 3):
            self._decode_verify_steps += 1
            for i in disk[:8]:
                print(f"[verify-decode] step={self._decode_verify_steps} "
                      f"row={int(src[i])} slot={int(slots[i])} ndisk={len(disk)}", flush=True)
                self._verify_slot(cache, layer_id,
                                  self._expert_of(int(src[i]), layer_id),
                                  int(slots[i]), phase="decode")
        disk_set = set(disk)
        ram = [i for i in range(n) if i not in disk_set]
        if ram:
            sel = torch.tensor(ram, dtype=torch.long)
            cache.src_indices[:len(ram)].copy_(src[sel].to(cache.src_indices.dtype))
            cache.evict_slots[:len(ram)].copy_(slots[sel].to(cache.evict_slots.dtype))
        cache.num_indices.fill_(len(ram))

    def refresh(self, cache) -> None:
        """Rebind the slot-cache references after a runtime cache rebuild."""
        self._banks = list(cache.banks)
        self._cache = cache

    def stats(self) -> dict:
        return {
            "experts_fetched": self._fetches,
            "bytes_fetched": self._fetch_bytes,
            "prefetch_hits": self._pf_hits,
            "prefetch_fetches": self._pf_fetches,
        }
