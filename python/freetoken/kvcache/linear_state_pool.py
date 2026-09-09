from __future__ import annotations

import math

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.utils import div_even, init_logger

logger = init_logger(__name__)

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Host-snapshot slot ids handed to the radix tree start here (GPU slot ids are
# always < 1 << 30). The pool routes free/copy on this bit.
_HOST_ID_BASE = 1 << 30


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads)`` for the GDN state tensors -- the single
    source of the sharding math shared by the pool allocation and the byte estimate."""
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
    return len(group.layer_ids), local_conv_dim, local_v_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Indexed by ``Req.table_idx`` (0..max_running_req), the same per-request slot the
    page table uses, so the scheduler's existing admit/free of ``table_idx`` covers the
    state's lifetime. One fixed slot per running request; no paging, no eviction.

    A model can declare extra per-request tensors on the same slots through
    ``ModelConfig.slot_states`` (see ``SlotStateSpec``); they advance, snapshot, COW and
    rebuild with the GDN state and are read back through ``slot_state(name, layer_id)``.
    Consumers must re-read them each forward: ``rebuild`` replaces the tensors.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
        slot_states: tuple[SlotStateSpec, ...] = (),
        snapshot_mode: str = "gpu",
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size
        if snapshot_mode not in ("gpu", "host"):
            raise ValueError(f"snapshot_mode {snapshot_mode!r} not in ('gpu', 'host')")

        self._group = group
        self._num_slots = num_slots
        self._device = device
        self._conv_dtype = dtype
        self._snapshot_mode = snapshot_mode

        # Host snapshot store (snapshot_mode="host"): tree-donated snapshots live in
        # pinned host RAM instead of VRAM slots, as lazily-allocated pinned buffers.
        # VRAM slots then only back the live working set (no ping-pong, no committed
        # snapshot, no snapshot cache) and the GDN VRAM budget shrinks accordingly
        # (state_pool_bytes -> _linear_pool_num_slots -> mr + 1).
        self._host_store: list[dict[str, torch.Tensor] | None] = []
        self._free_host: list[int] = []
        # The engine's forward stream; set by the engine via set_engine_stream. D2H
        # snapshots must be ordered after the forward writes that produced the state.
        self._engine_stream: torch.cuda.Stream | None = None

        n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}

        self._slot_specs = tuple(slot_states)
        names = [spec.name for spec in self._slot_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate slot_state names: {names}")
        self._state_layer_index = {
            spec.name: {lid: i for i, lid in enumerate(spec.layer_ids)}
            for spec in self._slot_specs
        }
        self.slot_states: dict[str, torch.Tensor] = self._alloc_slot_states(num_slots)

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as a padding sink,
        # sglang MambaPool convention). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single free-list, so memory
        # flows between them by demand. Unused by the op harness (which assigns slots by hand).
        self.padding_slot = 0
        self._free_slots: list[int] = list(range(1, num_slots))

    def _alloc_slot_states(self, num_slots: int) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.full(
                (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                spec.fill_value,
                dtype=spec.dtype if spec.dtype is not None else self._conv_dtype,
                device=self._device,
            )
            for spec in self._slot_specs
        }

    def has_slot_state(self, name: str) -> bool:
        return name in self.slot_states

    def slot_state(self, name: str, layer_id: int | None = None) -> torch.Tensor:
        """One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks the layer row."""
        t = self.slot_states[name]
        if layer_id is None:
            assert not self._state_layer_index[name], (
                f"slot_state {name!r} is per-layer, pass layer_id"
            )
            return t[0]
        return t[self._state_layer_index[name][layer_id]]

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a
        CacheManager rebuild that discards the tree owning donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice."""
        self._free_slots = list(range(1, self._num_slots))
        self._free_host = list(range(len(self._host_store)))

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only and destructive: every
        live/snapshot state is dropped, so the caller must guarantee no running request
        holds a slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = None
        self.recurrent_states = None
        self.slot_states = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        self.slot_states = self._alloc_slot_states(num_slots)
        self._num_slots = num_slots
        self._free_slots = list(range(1, num_slots))
        self._free_host = list(range(len(self._host_store)))

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor.
        Host snapshot ids (>= _HOST_ID_BASE) route to the host store."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        vram = []
        for s in slots:
            s = int(s)
            if s >= _HOST_ID_BASE:
                self._free_host_snapshot(s)
            else:
                vram.append(s)
        self._free_slots.extend(vram)

    # --- host snapshot store (snapshot_mode="host") ---

    @property
    def snapshot_mode(self) -> str:
        return self._snapshot_mode

    @property
    def host_snapshots(self) -> bool:
        return self._snapshot_mode == "host"

    def set_engine_stream(self, stream: torch.cuda.Stream | None) -> None:
        """Give the pool the engine's forward stream so snapshot_to_host can order its D2H
        after the forward writes even when called from the scheduler stream."""
        self._engine_stream = stream

    @property
    def num_host_snapshots(self) -> int:
        return len(self._host_store) - len(self._free_host)

    def host_snapshot_bytes(self) -> int:
        """Current pinned-RAM footprint of the host snapshot store."""
        if not self._host_store:
            return 0
        return int(sum(
            t.numel() * t.element_size()
            for e in self._host_store
            if e is not None
            for t in e.values()
        ))

    def is_host_snapshot(self, slot_id: int) -> bool:
        return slot_id >= _HOST_ID_BASE

    def _alloc_host_snapshot(self) -> tuple[int, dict[str, torch.Tensor]]:
        if self._free_host:
            idx = self._free_host.pop()
            return idx, self._host_store[idx]
        idx = len(self._host_store)
        entry = {
            "conv": torch.empty(
                self.conv_states[:, 0].shape,
                dtype=self.conv_states.dtype, device="cpu", pin_memory=True,
            ),
            "rec": torch.empty(
                self.recurrent_states[:, 0].shape,
                dtype=self.recurrent_states.dtype, device="cpu", pin_memory=True,
            ),
        }
        for spec in self._slot_specs:
            entry[spec.name] = torch.empty(
                self.slot_states[spec.name][:, 0].shape,
                dtype=self.slot_states[spec.name].dtype, device="cpu", pin_memory=True,
            )
        self._host_store.append(entry)
        return idx, entry

    def _free_host_snapshot(self, host_id: int) -> None:
        idx = host_id - _HOST_ID_BASE
        if not (0 <= idx < len(self._host_store)) or self._host_store[idx] is None:
            raise ValueError(f"invalid host snapshot id {host_id}")
        if idx in self._free_host:
            raise ValueError(f"host snapshot {host_id} freed twice")
        self._free_host.append(idx)

    def snapshot_to_host(self, slot: int) -> int:
        """D2H the whole-sequence state of VRAM ``slot`` into a pinned snapshot; returns the
        host id for the tree. The copy runs on the CALLER's current stream: cache_req is
        invoked from the scheduler loop after its cross-stream wait against the engine
        stream, so the source state is final, and every later writer of the slot or the
        host entry is enqueued on that same scheduler stream behind this copy (FIFO).
        No inline event waits: the scheduler thread drives both streams' waits, an inline
        synchronize here deadlocks (observed in overlap_loop wait_stream)."""
        idx, entry = self._alloc_host_snapshot()
        tensors = [("conv", entry["conv"], self.conv_states),
                   ("rec", entry["rec"], self.recurrent_states)]
        tensors += [(s.name, entry[s.name], self.slot_states[s.name]) for s in self._slot_specs]
        for _name, dst, src in tensors:
            # copy per leading layer: src[:, slot] is a strided view (non-contiguous), and
            # a whole-view D2H would materialize a contiguous staging copy on the GPU
            # (~60 MB on Qwen3.5 -- OOM on tight budgets). Per-layer tail slices are
            # contiguous views -> straight D2H, no staging.
            for l in range(src.shape[0]):
                dst[l].copy_(src[l, slot], non_blocking=True)
        return _HOST_ID_BASE + idx

    def restore_from_host(self, host_id: int, slot: int) -> None:
        """H2D a host snapshot back into VRAM ``slot`` (COW-restore of a prefix hit). Enqueued
        on the engine stream when the caller runs elsewhere (the engine stream consumes the
        live slot in its next forward, so same-stream ordering is what matters)."""
        idx = host_id - _HOST_ID_BASE
        if not (0 <= idx < len(self._host_store)):
            raise ValueError(f"invalid host snapshot id {host_id}")
        entry = self._host_store[idx]
        tensors = [("conv", self.conv_states, entry["conv"]),
                   ("rec", self.recurrent_states, entry["rec"])]
        tensors += [(s.name, self.slot_states[s.name], entry[s.name]) for s in self._slot_specs]

        for _name, dst, src in tensors:
            # per-LAYER copy: dst[:, slot] is a strided column view and an H2D
            # into it materializes a contiguous GPU staging buffer (~60 MB --
            # OOM on ~50 MB-free servers); dst[l, slot] tail slices are
            # contiguous views, so each layer copies staging-free.
            for l in range(dst.shape[0]):
                dst[l, slot].copy_(src[l], non_blocking=True)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, slots] = spec.fill_value

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot). Host
        snapshot ids route through the pinned store (H2D); both-VRAM stays D2D."""
        if src >= _HOST_ID_BASE:
            self.restore_from_host(src, dst)
            return
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, table_idx] = spec.fill_value

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        for t in self.slot_states.values():
            per += t[:, 0].numel() * t.element_size()
        return int(per)


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
    slot_states: tuple[SlotStateSpec, ...] = (),
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local), plus any
    declared slot_states."""
    n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    total = n_layers * (conv_bytes + rec_bytes)

    for spec in slot_states:
        item = (spec.dtype if spec.dtype is not None else dtype).itemsize
        total += max(1, len(spec.layer_ids)) * math.prod(spec.shape) * item
    return int(total)


__all__ = ["LinearStatePool", "linear_state_bytes_per_req"]


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup
    slot count). The engine adds this to the KV family's fixed cost when budgeting --
    the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    slot_states = getattr(config.model_config, "slot_states", ())
    if linear_group is None:
        if slot_states:
            raise ValueError("slot_states ride the linear-state slots; model has no linear group")
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    per_req = linear_state_bytes_per_req(
        linear_group, config.tp_info.size, config.dtype, slot_states
    )
    return per_req * slots


def _linear_pool_num_slots(config) -> int:
    """LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request
    (1 live + 2 ping-pong + 1 committed snapshot locked through decode), plus a cross-request
    snapshot cache and a padding sink; naive GDN keeps the old (max_running_req + 1). A
    linear_state_cache_ratio below 1.0 disables the snapshot cache entirely (floor: the
    non-evictable working set only). Host snapshot mode (--linear-state-snapshots host) moves
    tree snapshots to pinned RAM: the pool backs live + ping-pong only (2*mr + 1) -- the
    kernel still writes mid-chunk snapshots into the ping-pong slots, chunk commits donate
    HOST copies and hand the VRAM slot straight back."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    if getattr(config, "linear_state_snapshots", "gpu") == "host":
        # padding + live + 2 ping-pong per request; the tree never holds VRAM slots
        return 2 * mr + 2
    ratio = config.linear_state_cache_ratio
    if ratio < 1.0:
        return 4 * mr + 1  # zero snapshot cache (the _linear_pool_min_slots floor)
    n_cache = max(4, int(ratio * mr))
    return 4 * mr + n_cache + 1  # live + 2 ping-pong + locked committed snapshot + cache + padding


def _linear_pool_min_slots(config) -> int:
    """Floor on LinearStatePool slots that still runs: the non-evictable working set with a
    zero snapshot cache. Hybrid-radix needs 4 per running request (1 live + 2 ping-pong + 1
    committed snapshot locked through decode) + the padding sink; naive needs 1 per request +
    padding. Below this, a full max_running_req batch can't get its slots and admission
    deadlocks -- so a runtime rebuild rejects a smaller request."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    if getattr(config, "linear_state_snapshots", "gpu") == "host":
        return mr + 1
    return 4 * mr + 1
