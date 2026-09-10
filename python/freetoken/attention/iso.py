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




def chunked_decode(
    q: torch.Tensor,
    out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    scale: float,
    q_positions: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor],
    scratch_tokens: int,
    fmt: str,
) -> None:
    """Decode attention over a context LARGER than the dequant scratch.

    Splits the packed context into scratch-sized chunks; each chunk is
    dequantized into the shared scratch and served by the triton split-k
    decode kernel (one "split" per chunk), then the per-chunk partials are
    merged with the standard online-softmax (LSE) combination -- identical
    math to the kernel's own split-k combine. ``scratch`` holds
    (k, v) buffers of ``scratch_tokens`` rows each; rows are positional
    (scratch row j == global token chunk_start + j).
    """
    from freetoken.kernel.iso import iso_dequant_rows
    from freetoken.kernel.triton.attention import decode_paged_attention

    n, nq, d = q.shape
    total = int(kv_indptr[-1])
    ct = max(1, min(scratch_tokens, total))
    n_chunks = (total + ct - 1) // ct

    logits = torch.zeros(n, nq, n_chunks, d, dtype=torch.float32, device=q.device)
    lse = torch.full((n, nq, n_chunks), float("-inf"), dtype=torch.float32,
                     device=q.device)
    device = q.device
    idx_buf = torch.arange(ct, dtype=torch.int32, device=device)
    ones = torch.ones(n, dtype=torch.int32, device=device)
    sk, sv = scratch
    for c in range(n_chunks):
        lo, hi = c * ct, min((c + 1) * ct, total)
        size = hi - lo
        iso_dequant_rows(
            k_cache, v_cache, kv_indices[lo:hi], nheads_kv, head_dim, fmt,
            out=(sk[:size].view(size, -1), sv[:size].view(size, -1)),
        )
        indptr_c = (kv_indptr - lo).clamp_(0, size).to(torch.int32)
        decode_paged_attention(
            q=q,
            k_cache=sk[:size],
            v_cache=sv[:size],
            indptr=indptr_c,
            indices=idx_buf[:size],
            q_positions=q_positions,
            attn_logits=logits[:, :, c : c + 1],
            attn_lse=lse[:, :, c : c + 1],
            num_kv_splits=ones,
            max_kv_splits=1,
            sm_scale=scale,
        )

    m = lse.max(dim=-1).values                       # [n, nq]
    w = (lse - m.unsqueeze(-1)).exp()                # [n, nq, C]
    merged = (logits * w.unsqueeze(-1)).sum(dim=2) / w.sum(dim=-1)[..., None]
    out.copy_(merged.to(out.dtype).view(n, nq, d))


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



def iso_extend_chunked(
    q: torch.Tensor,
    out: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    prefix_indices: torch.Tensor,
    k_ext: torch.Tensor,
    v_ext: torch.Tensor,
    max_q_len: int,
    scale: float,
    fmt: str,
    scratch: tuple[torch.Tensor, torch.Tensor],
    indices_buf: torch.Tensor,
    lse_buf: torch.Tensor,
    ct: int,
) -> None:
    """Extend attention against a packed prefix too large for the one-shot
    scratch: the prefix is served in ``ct``-row chunks (each dequantized in
    place and attended with the non-causal triton extend kernel), plus one
    causal pass over the bf16 extend rows; the partials merge by LSE. The
    extend rows are re-attended once (empty-prefix split-kernel pass), not
    per chunk, so no softmax mass is double-counted.

    ``scratch`` is (k, v) bf16 [ct, kv_heads, head_dim] reused across layers;
    ``indices_buf`` is arange(ct, int32) (chunk rows are dense scratch ids);
    ``lse_buf`` is [n, nq] fp32 scratch. Writes ``out`` ([n, nq, head_dim] bf16).
    """
    from freetoken.kernel.iso import iso_dequant_rows
    from freetoken.kernel.triton.attention import extend_paged_attention

    sk, sv = scratch
    n, nq = q.shape[0], q.shape[1]
    pt = prefix_indices.numel()
    kv_heads, head_dim = sk.shape[1], sk.shape[2]
    R = kv_indptr.numel() - 1
    zero_indptr = torch.zeros(R + 1, dtype=torch.int32, device=q.device)
    zero_lens = torch.zeros(R, dtype=torch.int32, device=q.device)
    empty_idx = indices_buf[:0]
    acc = torch.zeros(n, nq, head_dim, dtype=torch.float32, device=q.device)
    lsum = torch.zeros(n, nq, dtype=torch.float32, device=q.device)
    run_max = torch.full((n, nq), float("-inf"), dtype=torch.float32, device=q.device)

    def merge(o_p: torch.Tensor, lse_p: torch.Tensor) -> None:
        nonlocal acc, lsum, run_max
        l_new = torch.maximum(run_max, lse_p)
        # both -inf -> weight 0 (avoids inf - inf = nan)
        w_old = torch.where(run_max == float("-inf"), 0.0, torch.exp(run_max - l_new))
        w_p = torch.where(lse_p == float("-inf"), 0.0, torch.exp(lse_p - l_new))
        acc.mul_(w_old.unsqueeze(-1)).add_(o_p.float() * w_p.unsqueeze(-1))
        lsum.mul_(w_old).add_(w_p)
        run_max = l_new

    for c0 in range(0, pt, ct):
        c1 = min(c0 + ct, pt)
        m = c1 - c0
        iso_dequant_rows(
            k_flat, v_flat, prefix_indices[c0:c1], kv_heads, head_dim, fmt,
            out=(sk[:m].view(m, -1), sv[:m].view(m, -1)),
        )
        chunk_indptr = (kv_indptr.clamp(c0, c1) - c0).to(torch.int32)
        chunk_lens = (chunk_indptr[1:] - chunk_indptr[:-1]).contiguous()
        o_p = extend_paged_attention(
            q=q, k_cache=sk[:m], v_cache=sv[:m], qo_indptr=cu_seqlens_q,
            kv_indptr=chunk_indptr, kv_indices=indices_buf[:m],
            prefix_lens=chunk_lens, max_q_len=max_q_len, sm_scale=scale,
            lse_out=lse_buf,
        )
        merge(o_p, lse_buf)

    # causal extend partial: empty prefix, split kernel handles the causal
    # bf16 extend rows only
    o_e = extend_paged_attention(
        q=q, k_cache=sk[:0], v_cache=sv[:0], qo_indptr=cu_seqlens_q,
        kv_indptr=zero_indptr, kv_indices=empty_idx, prefix_lens=zero_lens,
        max_q_len=max_q_len, sm_scale=scale,
        k_extend=k_ext, v_extend=v_ext, lse_out=lse_buf,
    )
    merge(o_e, lse_buf)
    out.copy_((acc / lsum.clamp_min(1e-30).unsqueeze(-1)).to(out.dtype))


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
        self._extend_lse: torch.Tensor | None = None
        self._decode_max_ctx = 0

    def _extend_chunk_rows(self, kv_heads: int, head_dim: int) -> int:
        """Chunk rows for the chunked extend path: live-VRAM sized (free +
        unused-reserved, minus 96 MB of triton workspace headroom), pinned by
        FREETOKEN_ISO_EXTEND_CT for debugging/tests. The caller falls back to
        the packed kernel below a 4096-row floor."""
        import os

        pinned = os.environ.get("FREETOKEN_ISO_EXTEND_CT")
        if pinned:
            return max(1, int(pinned))
        try:
            free_d, _total = torch.cuda.mem_get_info(self.device)
            cached = torch.cuda.memory_reserved(self.device) - \
                torch.cuda.memory_allocated(self.device)
        except Exception:
            return 0
        row = kv_heads * head_dim * 2 * 2  # K + V bf16
        return int((free_d + max(0, cached) - 96 * 2**20) // row)

    @staticmethod
    def _scratch_cap_bytes() -> int:
        """Upper bound for the transient bf16 prefix scratch (dequant for the
        triton extend path); beyond it the fallback CUDA extend kernel is used."""
        import os

        return int(os.environ.get("FREETOKEN_ISO_SCRATCH_MB", "128")) * 2**20

    @staticmethod
    def _scratch_tokens(kv_heads: int, head_dim: int) -> int:
        """How many context rows of dequantized K/V (bf16, K + V) fit in the
        scratch budget: the chunk size of the chunked decode path."""
        return max(1, IsoAttentionBackend._scratch_cap_bytes()
                   // (kv_heads * head_dim * 2 * 2))

    def _decode_small_context(self, q, k_flat, v_flat, metadata, kv_heads,
                              head_dim, scale, n, nq, total):
        """One-shot decode: the whole context fits the scratch budget."""
        from freetoken.kernel.iso import iso_dequant_rows
        from freetoken.kernel.triton.attention import decode_paged_attention

        if self._decode_scratch is None or self._decode_max_ctx < total:
            self._decode_scratch = (
                torch.empty(total, kv_heads, head_dim,
                            dtype=torch.bfloat16, device=self.device),
                torch.empty(total, kv_heads, head_dim,
                            dtype=torch.bfloat16, device=self.device),
            )
            self._decode_max_ctx = total
        sk, sv = self._decode_scratch
        iso_dequant_rows(
            k_flat, v_flat, metadata.indices, kv_heads, head_dim,
            self.iso_fmt,
            out=(sk[:total].view(total, -1), sv[:total].view(total, -1)),
        )
        if self._decode_indices is None or self._decode_indices.numel() < total:
            self._decode_indices = torch.arange(
                total, dtype=torch.int32, device=self.device)
        attn_logits = torch.empty(
            (n, nq, 8, head_dim), dtype=torch.float32, device=self.device)
        attn_lse = torch.empty((n, nq, 8), dtype=torch.float32, device=self.device)
        num_kv_splits = torch.full((n,), 8, dtype=torch.int32, device=self.device)
        return decode_paged_attention(
            q=q,
            k_cache=sk[:total],
            v_cache=sv[:total],
            indptr=metadata.indptr,
            indices=self._decode_indices[:total],
            q_positions=metadata.q_positions,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=8,
            sm_scale=scale,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        from freetoken.kernel.iso import iso_attention_decode, iso_attention_extend  # noqa: F401

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
            # capture sessions pin buffer shapes at capture time and indptr[-1]
            # is a D2H read (illegal inside a graph) -> graphs always take the
            # packed kernel path
            if self.capture is None and n == 1:
                total = int(metadata.indptr[-1])
                if total > 0:
                    free_bytes, _ = torch.cuda.mem_get_info(self.device)
                    # driver free alone lies after a big prefill: the caching
                    # allocator holds the transient blocks as reserved-unused
                    # (same lesson as moe's _prefill_block_size)
                    cached = torch.cuda.memory_reserved(self.device) - \
                        torch.cuda.memory_allocated(self.device)
                    free_bytes += max(0, cached)
                    per_token = kv_heads * head_dim * 2 * 2
                    if self._decode_scratch is not None:
                        # resident scratch is already paid for: reuse its full
                        # size, no VRAM-dependent shrink
                        ct = max(1, min(self._decode_max_ctx, total))
                        if free_bytes > 16 * 2**20:
                            if ct >= total:
                                return self._decode_small_context(
                                    q, k_flat, v_flat, metadata, kv_heads,
                                    head_dim, scale, n, nq, total)
                            out = torch.empty_like(q)
                            chunked_decode(
                                q, out, k_flat, v_flat, metadata.indptr,
                                metadata.indices, nq, kv_heads, head_dim,
                                scale, metadata.q_positions,
                                self._decode_scratch, ct, self.iso_fmt,
                            )
                            return out
                    else:
                        # first allocation: size the scratch from live VRAM
                        # (free - 48 MB, halved as growth headroom). A floor of
                        # 4096 rows keeps the chunk count (and the [n, heads,
                        # chunks, D] partial buffers) sane; below that the
                        # packed kernel runs (correct, slower).
                        ct_env = self._scratch_tokens(kv_heads, head_dim)
                        ct_live = (free_bytes - 48 * 2**20) // 2 // per_token
                        ct = min(ct_env, ct_live, total)
                        if ct < 4096:
                            out = torch.empty_like(q)
                            iso_attention_decode(
                                q.reshape(n, -1), out.reshape(n, -1), k_flat,
                                v_flat, metadata.indptr, metadata.indices,
                                nq, kv_heads, head_dim, scale, self.iso_fmt,
                            )
                            return out
                        scratch_bytes = ct * per_token
                        if free_bytes > scratch_bytes + 48 * 2**20:
                            self._decode_scratch = (
                                torch.empty(ct, kv_heads, head_dim,
                                            dtype=torch.bfloat16, device=self.device),
                                torch.empty(ct, kv_heads, head_dim,
                                            dtype=torch.bfloat16, device=self.device),
                            )
                            self._decode_max_ctx = ct
                            if ct >= total:
                                return self._decode_small_context(
                                    q, k_flat, v_flat, metadata, kv_heads,
                                    head_dim, scale, n, nq, total)
                            out = torch.empty_like(q)
                            chunked_decode(
                                q, out, k_flat, v_flat, metadata.indptr,
                                metadata.indices, nq, kv_heads, head_dim,
                                scale, metadata.q_positions,
                                self._decode_scratch, ct, self.iso_fmt,
                            )
                            return out
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
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        # same live-VRAM guard as the decode path: the dequant scratch + triton
        # workspace must fit with headroom (2x scratch + 64 MB), otherwise the
        # packed CUDA extend kernel runs (no scratch at all)
        fast_ok = (
            prefix_total > 0
            and scratch_bytes <= self._scratch_cap_bytes()
            and free_bytes > 2 * scratch_bytes + 64 * 2**20
        )
        if fast_ok:
            from freetoken.kernel.iso import iso_dequant_rows
            from freetoken.kernel.triton.attention import extend_paged_attention

            # grow the reused scratch (shared with the decode path) and
            # dequantize STRAIGHT into it -- no transient kd/vd allocations
            if self._decode_scratch is None or self._decode_max_ctx < prefix_total:
                self._decode_scratch = (
                    torch.empty(prefix_total, kv_heads, head_dim,
                                dtype=torch.bfloat16, device=self.device),
                    torch.empty(prefix_total, kv_heads, head_dim,
                                dtype=torch.bfloat16, device=self.device),
                )
                self._decode_max_ctx = prefix_total
            sk, sv = self._decode_scratch
            iso_dequant_rows(
                k_flat, v_flat, metadata.prefix_indices, kv_heads, head_dim,
                self.iso_fmt,
                out=(sk[:prefix_total].view(prefix_total, -1),
                     sv[:prefix_total].view(prefix_total, -1)),
            )
            out = extend_paged_attention(
                q=q,
                k_cache=sk[:prefix_total],
                v_cache=sv[:prefix_total],
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
            # chunked: prefix slices through a fixed scratch (VRAM-sized, not
            # prefix-sized), each served by the triton extend kernel, partials
            # merged by LSE -- flat rate at any prefix length. Falls back to
            # the packed CUDA extend kernel only when even one chunk does not
            # fit the live VRAM.
            ct = self._extend_chunk_rows(kv_heads, head_dim)
            if ct >= 4096:
                if self._decode_scratch is None or self._decode_max_ctx < ct:
                    self._decode_scratch = (
                        torch.empty(ct, kv_heads, head_dim,
                                    dtype=torch.bfloat16, device=self.device),
                        torch.empty(ct, kv_heads, head_dim,
                                    dtype=torch.bfloat16, device=self.device),
                    )
                    self._decode_max_ctx = ct
                if self._decode_indices is None or self._decode_indices.numel() < ct:
                    self._decode_indices = torch.arange(
                        ct, dtype=torch.int32, device=self.device)
                if self._extend_lse is None:
                    self._extend_lse = torch.empty(
                        n, nq, dtype=torch.float32, device=self.device)
                out = torch.empty_like(q)
                iso_extend_chunked(
                    q, out, k_flat, v_flat, metadata.cu_seqlens_q_gpu,
                    metadata.kv_indptr, metadata.prefix_indices,
                    k.reshape(n, kv_heads, head_dim),
                    v.reshape(n, kv_heads, head_dim),
                    metadata.max_q_len, scale, self.iso_fmt,
                    self._decode_scratch, self._decode_indices,
                    self._extend_lse, ct,
                )
            else:
                out = torch.empty_like(q)
                iso_attention_extend(
                    q.reshape(n, -1), out.reshape(n, -1), k_flat, v_flat,
                    k.reshape(n, -1), v.reshape(n, -1),
                    metadata.cu_seqlens_q_gpu, metadata.prefix_indptr,
                    metadata.prefix_indices,
                    nq, kv_heads, head_dim, scale, metadata.max_q_len,
                    self.iso_fmt,
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
