"""Fused EXL3 dequant kernel tests (kernel/csrc/jit/exl3_dequant.cu).

Oracle: freetoken.models.exl3.dequant_exl3 (torch reference, bit-exact vs the
exllamav3 CUDA implementation). Any uint16 payload is a valid trellis, so
random payloads + random fp16 scales give full decode-path coverage.
"""

import pytest
import torch

from freetoken.models.exl3 import dequant_exl3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@pytest.mark.parametrize("k_bits", [2, 3, 4])
@pytest.mark.parametrize("codebook", [0, 1, 2])
@pytest.mark.parametrize("in_dim,out_dim", [(256, 128), (128, 256)])
def test_fused_dequant_vs_reference(k_bits, codebook, in_dim, out_dim):
    from freetoken.kernel.exl3 import exl3_dequant_fused

    torch.manual_seed(17 + k_bits * 10 + codebook + in_dim)
    dev = "cuda"
    m = 3
    tb = in_dim * out_dim * k_bits // 8
    suh_b = 2 * in_dim
    svh_b = 2 * out_dim
    row_bytes = tb + suh_b + svh_b

    rows = torch.randint(0, 256, (m, row_bytes), dtype=torch.uint8, device=dev)
    # random signed scales, fp16-representable magnitudes
    for j in range(m):
        rows[j, tb : tb + suh_b] = (
            (torch.randn(in_dim, device=dev).half().view(torch.uint8))
        )
        rows[j, tb + suh_b :] = (
            (torch.randn(out_dim, device=dev).half().view(torch.uint8))
        )
    out = torch.empty(m, out_dim, in_dim, dtype=torch.bfloat16, device=dev)
    exl3_dequant_fused(rows, out, 0, tb, tb + suh_b, in_dim, out_dim, codebook, k_bits)
    torch.cuda.synchronize()

    trellis = (
        rows[:, :tb]
        .view(torch.int16)
        .reshape(m, in_dim // 16, out_dim // 16, 16 * k_bits)
    )
    suh = rows[:, tb : tb + suh_b].view(torch.float16).reshape(m, in_dim)
    svh = rows[:, tb + suh_b :].view(torch.float16).reshape(m, out_dim)
    ref = dequant_exl3(trellis, suh, svh, codebook)  # fp32 [m, out, in]

    got = out.float()
    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(m, -1), got.reshape(m, -1), dim=-1
    )
    assert cos.min() > 0.9999, f"cos {cos.min()}"
    rel = (got - ref).norm(dim=(1, 2)) / ref.norm(dim=(1, 2)).clamp_min(1e-12)
    assert rel.max() < 2e-3, f"rel L2 {rel.max()}"


@pytest.mark.parametrize("k_bits", [2, 3, 4])
@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_fused_rows_layout(k_bits, kind):
    """_dequant_rows (fused dispatch) vs the torch reference path, with the real
    padded bank-row layout (max-K field width) and non-contiguous out views."""
    from freetoken.moe.fused_exl3 import _dequant_rows, _dequant_rows_torch

    torch.manual_seed(29 + k_bits)
    dev = "cuda"
    H, I = 256, 128
    m, cb = 3, 1
    k_max = 4  # rows padded to the checkpoint max-K layout
    tb_max = H * I * k_max // 8
    row_bytes = (2 * tb_max + 4 * H + 4 * I) if kind == "gate_up" else (tb_max + 2 * H + 2 * I)
    rows = torch.randint(0, 256, (m, row_bytes), dtype=torch.uint8, device=dev)

    def _scales(off_h, off_v, nh, nv):
        rows[:, off_h : off_h + 2 * nh] = torch.randn(m, nh, device=dev).half().view(torch.uint8)
        rows[:, off_v : off_v + 2 * nv] = torch.randn(m, nv, device=dev).half().view(torch.uint8)

    if kind == "gate_up":
        _scales(2 * tb_max, 2 * tb_max + 2 * H, H, I)
        _scales(2 * tb_max + 2 * H + 2 * I, 2 * tb_max + 4 * H + 2 * I, H, I)
    else:
        _scales(tb_max, tb_max + 2 * I, I, H)

    got = _dequant_rows(rows, H, I, kind, cb, k_bits)
    ref = _dequant_rows_torch(rows, H, I, kind, cb, k_bits)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().reshape(m, -1), got.float().reshape(m, -1), dim=-1
    )
    assert cos.min() > 0.9999, f"{kind} K={k_bits}: cos {cos.min()}"
