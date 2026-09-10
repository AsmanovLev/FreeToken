"""Chunked extend tests (attention/iso.py iso_extend_chunked).

Oracle: torch attention over python-reference-dequantized packed prefix +
causal bf16 extend rows. Exercises ragged per-request prefixes so the
clamp-based per-chunk indptr arithmetic is covered.
"""

import pytest
import torch

from freetoken.attention.iso import iso_extend_chunked
from freetoken.kernel import iso

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _run(fmt, ct, ctxs, n_new, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    HQ, HKV = 16, 2
    total = sum(ctxs)
    slots = total + 8
    rb = iso.packed_row_bytes(D := 256, fmt)
    kc = torch.zeros(slots, HKV * rb, dtype=torch.uint8, device=dev)
    vc = torch.zeros_like(kc)
    k = (torch.randn(total, HKV, D, device=dev) * 2).to(torch.bfloat16)
    v = (torch.randn(total, HKV, D, device=dev) * 2).to(torch.bfloat16)
    pidx = torch.randperm(slots, device=dev, dtype=torch.int32)[:total]
    iso.iso_store_cache(kc, vc, pidx, k.view(total, -1), v.view(total, -1), HKV, D, fmt)
    torch.cuda.synchronize()

    ke = (torch.randn(n_new, HKV, D, device=dev) * 2).to(torch.bfloat16)
    ve = (torch.randn(n_new, HKV, D, device=dev) * 2).to(torch.bfloat16)
    q = torch.randn(n_new, HQ, D, device=dev).to(torch.bfloat16)
    cu_q = torch.arange(0, len(ctxs) + 1, dtype=torch.int32, device=dev) * (n_new // len(ctxs))
    cu_q[-1] = n_new
    kv_indptr = torch.tensor([0] + list(torch.tensor(ctxs).cumsum(0)), dtype=torch.int32, device=dev)
    scale = D ** -0.5

    scratch = (torch.empty(ct, HKV, D, dtype=torch.bfloat16, device=dev),
               torch.empty(ct, HKV, D, dtype=torch.bfloat16, device=dev))
    indices_buf = torch.arange(ct, dtype=torch.int32, device=dev)
    lse_buf = torch.empty(n_new, HQ, dtype=torch.float32, device=dev)
    out = torch.empty_like(q)
    iso_extend_chunked(q, out, kc.view(-1, HKV * rb), vc.view(-1, HKV * rb),
                       cu_q, kv_indptr, pidx, ke, ve,
                       n_new // len(ctxs), scale, fmt, scratch, indices_buf,
                       lse_buf, ct)
    torch.cuda.synchronize()

    kd, vd = iso.iso_dequant_rows(kc, vc, pidx, HKV, D, fmt)
    kd = kd.float().view(total, HKV, D)
    vd = vd.float().view(total, HKV, D)
    gq = HQ // HKV
    for r, ctx in enumerate(ctxs):
        qs = int(cu_q[r])
        n = int(cu_q[r + 1]) - qs
        ks = int(kv_indptr[r])
        kk = torch.cat([kd[ks : ks + ctx].repeat_interleave(gq, 1),
                        ke[qs : qs + n].float().repeat_interleave(gq, 1)])
        vv = torch.cat([vd[ks : ks + ctx].repeat_interleave(gq, 1),
                        ve[qs : qs + n].float().repeat_interleave(gq, 1)])
        sc = torch.einsum("nhd,thd->nht", q[qs : qs + n].float(), kk) * scale
        allow = (torch.arange(ctx + n, device=dev).unsqueeze(0)
                 < ctx + torch.arange(1, n + 1, device=dev).unsqueeze(1))
        sc = sc.masked_fill(~allow.unsqueeze(1), float("-inf"))
        ref = torch.einsum("nht,thd->nhd", sc.softmax(-1), vv)
        got = out[qs : qs + n].float()
        cos = torch.nn.functional.cosine_similarity(ref.reshape(-1), got.reshape(-1), dim=0)
        assert cos > 0.999, f"req{r} cos={cos}"


@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
def test_single_request_multi_chunk(fmt):
    _run(fmt, ct=1024, ctxs=[3000], n_new=40, seed=0)


def test_ragged_requests_span_chunks():
    # two requests whose prefixes straddle chunk boundaries (clamp path)
    _run("iso3", ct=1024, ctxs=[1500, 2500], n_new=8, seed=1)


def test_single_chunk_equals_one_shot():
    # ct >= prefix: one chunk + causal extend partial
    _run("iso4", ct=4096, ctxs=[1200], n_new=16, seed=2)
