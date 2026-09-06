"""EXL3 bank row packing + on-the-fly dequant tests (moe/fused_exl3.py)."""

import pytest
import torch

from freetoken.models.exl3 import dequant_exl3
from freetoken.moe.fused_exl3 import _dequant_rows, _row_params

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _pack_gu(gate_t, up_t, gate_suh, gate_svh, up_suh, up_svh, tb_max):
    def pad(t):
        b = t.view(torch.uint8).flatten()
        return torch.cat([b, torch.zeros(tb_max - b.numel(), dtype=torch.uint8)])

    return torch.cat(
        [
            pad(gate_t),
            pad(up_t),
            gate_suh.view(torch.uint8),
            gate_svh.view(torch.uint8),
            up_suh.view(torch.uint8),
            up_svh.view(torch.uint8),
        ]
    )


def _pack_dn(t, suh, svh, tb_max):
    b = t.view(torch.uint8).flatten()
    return torch.cat(
        [torch.cat([b, torch.zeros(tb_max - b.numel(), dtype=torch.uint8)]),
         suh.view(torch.uint8), svh.view(torch.uint8)]
    )


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_row_dequant_roundtrip(cb):
    """Random trellis bytes pack -> _dequant_rows == direct dequant_exl3."""
    torch.manual_seed(17)
    H, I, K = 256, 128, 3
    n = 4
    dev = "cuda"
    gate_t = torch.randint(-32768, 32767, (n, H // 16, I // 16, 16 * K), dtype=torch.int16)
    up_t = torch.randint(-32768, 32767, (n, H // 16, I // 16, 16 * K), dtype=torch.int16)
    down_t = torch.randint(-32768, 32767, (n, I // 16, H // 16, 16 * K), dtype=torch.int16)
    gate_suh = torch.randn(n, H, dtype=torch.float16)
    gate_svh = torch.randn(n, I, dtype=torch.float16)
    up_suh = torch.randn(n, H, dtype=torch.float16)
    up_svh = torch.randn(n, I, dtype=torch.float16)
    down_suh = torch.randn(n, I, dtype=torch.float16)
    down_svh = torch.randn(n, H, dtype=torch.float16)

    tb_max = H * I * (K + 1) // 8  # simulate a checkpoint whose max K is K+1
    rows_gu = torch.stack(
        [_pack_gu(gate_t[i], up_t[i], gate_suh[i], gate_svh[i], up_suh[i], up_svh[i], tb_max)
         for i in range(n)]
    ).cuda()
    rows_dn = torch.stack(
        [_pack_dn(down_t[i], down_suh[i], down_svh[i], tb_max) for i in range(n)]
    ).cuda()

    w1 = _dequant_rows(rows_gu, H, I, "gate_up", cb, K)
    w2 = _dequant_rows(rows_dn, H, I, "down", cb, K)
    assert w1.shape == (n, 2 * I, H) and w1.dtype == torch.bfloat16
    assert w2.shape == (n, H, I)

    ref_gate = dequant_exl3(gate_t.cuda(), gate_suh.cuda(), gate_svh.cuda(), cb)  # [n, I, H] fp32
    ref_up = dequant_exl3(up_t.cuda(), up_suh.cuda(), up_svh.cuda(), cb)
    ref_down = dequant_exl3(down_t.cuda(), down_suh.cuda(), down_svh.cuda(), cb)  # [n, H, I]

    # per-expert dequant loop vs the batched reference: cuBLAS may pick a
    # different fp32 matmul kernel per batch size, so allow 1 bf16 ulp
    def close(a, b):
        a, b = a.float().cpu(), b.to(torch.bfloat16).float().cpu()
        return torch.allclose(a, b, rtol=2**-8, atol=1e-2)

    assert close(w1[:, :I], ref_gate)
    assert close(w1[:, I:], ref_up)
    assert close(w2, ref_down)


def test_row_params():
    H, I, K = 2048, 512, 3
    tb = H * I * K // 8
    tb_max = H * I * 4 // 8  # max-K layout
    assert _row_params(2 * tb_max + 4 * H + 4 * I, H, I, "gate_up", K) == (tb, tb_max)
    assert _row_params(tb_max + 2 * H + 2 * I, H, I, "down", K) == (tb, tb_max)
    with pytest.raises(ValueError):
        _row_params(12345, H, I, "down", K)
