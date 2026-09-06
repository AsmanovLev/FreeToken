"""ISO attention backend: full attention over ISOKVCache (ISO3/ISO4 packed KV).

Decode: quantize-on-write (store_kv packs the new token), then the packed
paged decode kernel. Extend (prefill): attention reads the packed prefix plus
the bf16 extend rows, and the new tokens are packed into the pool AFTER the
attention pass (deferred quantization — prefill never consumes its own
quantized K/V).

No sliding window / attention sinks (plain FULL attention only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from freetoken.models import ModelConfig




@dataclass
class IsoCaptureData(BaseCaptureData):
    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


@dataclass
class IsoMetadata(BaseAttnMetadata):
    cu_seqlens_q_gpu: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    prefix_indptr: torch.Tensor
    prefix_indices: torch.Tensor
    prefix_lens: torch.Tensor
    scratch_indices: torch.Tensor
    prefix_total: int
    is_decode: bool
    max_q_len: int
    q_positions: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class IsoAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.capture: IsoCaptureData | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))
        kv_groups = getattr(config, "kv_cache_group_specs", lambda: ())()
        self.max_head_dim = max(
            (group.head_dim for group in kv_groups),
            default=int(getattr(config, "head_dim", 1)),
        )
        self.iso_fmt = getattr(self.kvcache, "iso_fmt", "iso3")
        # dense bf16 decode scratch (k, v) grown on demand; reused across layers
        self._decode_scratch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._decode_indices: torch.Tensor | None = None
        self._decode_max_ctx = 0

    @staticmethod
    def _scratch_cap_bytes() -> int:
        """Upper bound for the transient bf16 prefix scratch (dequant for the
        triton extend path); beyond it the fallback CUDA extend kernel is used."""
        import os

        return int(os.environ.get("FREETOKEN_ISO_SCRATCH_MB", "128")) * 2**20

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        from freetoken.kernel.iso import iso_attention_decode, iso_attention_extend

        metadata = batch.attn_metadata
        assert isinstance(metadata, IsoMetadata)
        spec = attn_spec or AttentionSpec()
        if spec.sliding_window is not None or spec.sinks is not None:
            raise NotImplementedError(
                "iso attention backend does not support sliding windows or sinks"
            )
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5

        k_raw = self.kvcache.k_cache(layer_id)  # [pages, ps, heads, row_bytes] uint8
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads = k_raw.shape[-2]
        head_dim = q.shape[-1]
        k_flat = k_raw.view(-1, k_raw.shape[-2] * k_raw.shape[-1])
        v_flat = v_raw.view(-1, v_raw.shape[-2] * v_raw.shape[-1])

        n = q.shape[0]
        nq = q.shape[1]
        if metadata.is_decode:
            # quantize-on-write, then attend the whole sequence. The packed pool
            # is dequantized ONCE per layer into a dense bf16 scratch and fed to
            # the stock triton flash-decode kernel: O(ctx) with a small constant,
            # ~90x faster at 32k than the packed custom decode kernel.
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
            total = int(metadata.indptr[-1])
            scratch_bytes = total * kv_heads * head_dim * 2 * 2
            # capture sessions pin buffer shapes at capture time and indptr[-1]
            # is a D2H read (illegal inside a graph) -> graphs always take the
            # packed kernel path
            free_ok = False
            if self.capture is None and total > 0 and scratch_bytes <= self._scratch_cap_bytes():
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                free_ok = free_bytes > scratch_bytes + 64 * 2**20
            if free_ok:
                from freetoken.kernel.iso import iso_dequant_rows
                from freetoken.kernel.triton.attention import decode_paged_attention

                kd, vd = iso_dequant_rows(
                    k_flat, v_flat, metadata.indices, kv_heads, head_dim,
                    self.iso_fmt,
                )
                total_kv, max_splits = total, 8
                if self._decode_scratch is None or self._decode_max_ctx < total_kv:
                    self._decode_scratch = (
                        torch.empty(total_kv, kv_heads, head_dim,
                                    dtype=torch.bfloat16, device=self.device),
                        torch.empty(total_kv, kv_heads, head_dim,
                                    dtype=torch.bfloat16, device=self.device),
                    )
                    self._decode_max_ctx = total_kv
                assert self._decode_scratch is not None
                self._decode_scratch[0][:total_kv].copy_(kd.view(total_kv, kv_heads, head_dim))
                self._decode_scratch[1][:total_kv].copy_(vd.view(total_kv, kv_heads, head_dim))
                sk, sv = self._decode_scratch
                if self._decode_indices is None or self._decode_indices.numel() < total_kv:
                    self._decode_indices = torch.arange(
                        total_kv, dtype=torch.int32, device=self.device)
                attn_logits = torch.empty(
                    (n, nq, 8, head_dim), dtype=torch.float32, device=self.device)
                attn_lse = torch.empty((n, nq, 8), dtype=torch.float32, device=self.device)
                num_kv_splits = torch.full((n,), 8, dtype=torch.int32, device=self.device)
                return decode_paged_attention(
                    q=q,
                    k_cache=sk[:total_kv],
                    v_cache=sv[:total_kv],
                    indptr=metadata.indptr,
                    indices=self._decode_indices[:total_kv],
                    q_positions=metadata.q_positions,
                    attn_logits=attn_logits,
                    attn_lse=attn_lse,
                    num_kv_splits=num_kv_splits,
                    max_kv_splits=8,
                    sm_scale=scale,
                )
            out = torch.empty_like(q)
            iso_attention_decode(
                q.reshape(n, -1), out.reshape(n, -1), k_flat, v_flat,
                metadata.indptr, metadata.indices,
                nq, kv_heads, head_dim, scale, self.iso_fmt,
            )
            return out

        # extend: attend packed prefix + bf16 extend rows, THEN pack new tokens
        # (deferred quantization). The packed prefix is dequantized ONCE into a
        # dense bf16 scratch and fed to the regular tiled triton extend kernel —
        # O(prefix) dequant per layer instead of per-query dequant.
        prefix_total = metadata.prefix_total
        scratch_bytes = prefix_total * kv_heads * head_dim * 2 * 2
        if prefix_total > 0 and scratch_bytes <= self._scratch_cap_bytes():
            from freetoken.kernel.iso import iso_dequant_rows
            from freetoken.kernel.triton.attention import extend_paged_attention

            kd, vd = iso_dequant_rows(
                k_flat, v_flat, metadata.prefix_indices, kv_heads, head_dim,
                self.iso_fmt,
            )
            out = extend_paged_attention(
                q=q,
                k_cache=kd.view(prefix_total, kv_heads, head_dim),
                v_cache=vd.view(prefix_total, kv_heads, head_dim),
                qo_indptr=metadata.cu_seqlens_q_gpu,
                kv_indptr=metadata.indptr,
                kv_indices=metadata.scratch_indices,
                prefix_lens=metadata.prefix_lens,
                max_q_len=metadata.max_q_len,
                sm_scale=scale,
                k_extend=k.reshape(n, kv_heads, head_dim),
                v_extend=v.reshape(n, kv_heads, head_dim),
            )
        else:
            out = torch.empty_like(q)
            iso_attention_extend(
                q.reshape(n, -1), out.reshape(n, -1), k_flat, v_flat,
                k.reshape(n, -1), v.reshape(n, -1),
                metadata.cu_seqlens_q_gpu, metadata.prefix_indptr,
                metadata.prefix_indices,
                nq, kv_heads, head_dim, scale, metadata.max_q_len, self.iso_fmt,
            )
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return out

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = self.device
        page_table = get_global_ctx().page_table
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        is_decode = max(seqlens_q) == 1

        indptr = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum_(0)
        if is_decode:
            cu_seqlens_q_gpu = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        else:
            cu_seqlens_q_gpu = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=device
            ).cumsum_(0)
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        if is_decode:
            # decode attends the packed pool only; prefix split stays unused
            prefix_indptr = cu_seqlens_q_gpu
            prefix_indices = indices[:0]
            prefix_lens = torch.zeros(len(reqs), dtype=torch.int32, device=device)
            scratch_indices = indices
            prefix_total = 0
        else:
            prefix_indptr = torch.tensor(
                [0] + cached_lens, dtype=torch.int32, device=device
            ).cumsum_(0)
            prefix_indices = torch.cat(
                [page_table[req.table_idx, : req.cached_len] for req in reqs]
            )
            prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
            prefix_total = sum(cached_lens)
            # triton extend path: the dequantized prefix lives in a DENSE scratch
            # buffer, so per request the first prefix_len entries of kv_indices are
            # scratch row ids (extend part is read from k_extend, entries unused).
            parts = []
            off = 0
            for req in reqs:
                plen, elen = req.cached_len, req.extend_len
                parts.append(torch.arange(off, off + plen, dtype=torch.int32, device=device))
                parts.append(torch.zeros(elen, dtype=torch.int32, device=device))
                off += plen
            scratch_indices = (
                torch.cat(parts)
                if parts
                else torch.empty(0, dtype=torch.int32, device=device)
            )

        if is_decode:
            # triton decode expects the query token position per request
            q_positions = torch.tensor(
                [dl - 1 for dl in seqlens_k], dtype=torch.int32, device=device
            )
        else:
            q_positions = getattr(batch, "positions", None)

        batch.attn_metadata = IsoMetadata(
            cu_seqlens_q_gpu=cu_seqlens_q_gpu,
            indptr=indptr,
            indices=indices,
            prefix_indptr=prefix_indptr,
            prefix_indices=prefix_indices,
            prefix_lens=prefix_lens,
            scratch_indices=scratch_indices,
            prefix_total=prefix_total,
            is_decode=is_decode,
            max_q_len=max(seqlens_q),
            q_positions=q_positions,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = IsoCaptureData.create(max_bs, max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max_bs

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        capture = self.capture
        batch.attn_metadata = IsoMetadata(
            cu_seqlens_q_gpu=capture.cu_seqlens_q[: bs + 1],
            indptr=capture.cu_seqlens_k[: bs + 1],
            indices=capture.page_table.view(-1),
            prefix_indptr=capture.cu_seqlens_k[: bs + 1],
            prefix_indices=capture.page_table.view(-1),
            prefix_lens=capture.seq_lens[:bs],
            scratch_indices=capture.page_table.view(-1),
            prefix_total=0,
            is_decode=True,
            max_q_len=1,
            q_positions=capture.positions[:bs],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, IsoMetadata)
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture
        capture.cu_seqlens_q[: bs + 1].copy_(metadata.cu_seqlens_q_gpu)
        capture.cu_seqlens_k[: bs + 1].copy_(metadata.indptr)
        indices = capture.page_table.view(-1)
        total = metadata.indices.numel()
        indices[:total].copy_(metadata.indices)
        if metadata.q_positions is not None:
            capture.positions[: metadata.q_positions.numel()].copy_(metadata.q_positions)
            metadata.q_positions = capture.positions[: metadata.q_positions.numel()]
        metadata.cu_seqlens_q_gpu = capture.cu_seqlens_q[: bs + 1]
        metadata.indptr = capture.cu_seqlens_k[: bs + 1]
        metadata.indices = indices
