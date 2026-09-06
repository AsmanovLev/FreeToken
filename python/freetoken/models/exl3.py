"""EXL3 (QTIP trellis) weight dequantization, torch reference implementation.

Ported from exllamav3 (exllamav3_ext/quant/{pack.cu,exl3_dq.cuh,codebook.cuh},
modules/quant/exl3_lib/quantize.py). EXL3 stores a linear layer W (in, out) as:

  {key}.trellis  int16  (in/16, out/16, 16*K)  packed K-bit codes (K = bits/weight)
  {key}.suh      fp16   (in,)                  signed input scales (sign flip x scale)
  {key}.svh      fp16   (out,)                 signed output scales
  {key}.mcg / .mul1   int32 scalar sentinel    codebook selector (absent = legacy)
  {key}.bias     fp16   (out,)                 optional

Decode (no Viterbi at inference; the trellis search happened at quantize time):
  1. per 16x16 tile, extract a 16-bit tail-biting window per element:
     window(e) low K bits = code[e], upper bits = previous elements' codes;
  2. value = codebook_decode(window)  (procedural QTIP codebook, fp16-exact);
  3. place into the tile in tensor-core fragment order (tensor_core_perm);
  4. un-rotate: block-diagonal normalized Sylvester Hadamard H128 on the left
     (over input dim) and right (over output dim);
  5. W = W * suh[:, None] * svh[None, :]; HF orientation is W.T [out, in].
"""

from __future__ import annotations

import functools

import torch

EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".su", ".sv", ".mcg", ".mul1")

_HAD_K = 128

# fp16 bit-pattern constants of the mul1 codebook epilogue
_MUL1_K_INV = 0.00676727294921875  # fp16 bits 0x1EEE
_MUL1_K_BIAS = -10.3828125  # fp16 bits 0xC931


def is_exl3_layer(keyset, prefix: str) -> bool:
    """exllamav3 Linear.is_exl3_storage: trellis + (su|suh) + (sv|svh)."""
    return (
        f"{prefix}.trellis" in keyset
        and (f"{prefix}.su" in keyset or f"{prefix}.suh" in keyset)
        and (f"{prefix}.sv" in keyset or f"{prefix}.svh" in keyset)
    )


def codebook_of(keyset, prefix: str) -> int:
    """2 = mul1, 1 = mcg, 0 = legacy 3inst."""
    if f"{prefix}.mul1" in keyset:
        return 2
    if f"{prefix}.mcg" in keyset:
        return 1
    return 0


@functools.cache
def _tensor_core_perm(device: torch.device) -> torch.Tensor:
    """perm[j] = row-major flat tile index (r*16+c) of bitstream element j."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 : t * 8 + 8] = [
            r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
            r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1,
        ]
    return torch.tensor(perm, dtype=torch.long, device=device)


@functools.cache
def _hadamard128(device: torch.device) -> torch.Tensor:
    """Normalized Sylvester Hadamard H128 / sqrt(128) (symmetric, orthonormal)."""
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < _HAD_K:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h * (_HAD_K ** -0.5)


def _fp16_from_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int16).view(torch.float16)


def _decode_windows(win: torch.Tensor, codebook: int) -> torch.Tensor:
    """win: (..., ) int64 window values [0, 2^16) -> fp32 grid values (fp16-exact)."""
    if codebook == 2:  # mul1
        x = (win * 0x83DCD12D) & 0xFFFFFFFF
        s = (
            (x & 0xFF)
            + ((x >> 8) & 0xFF)
            + ((x >> 16) & 0xFF)
            + ((x >> 24) & 0xFF)
            + 0x6400
        )
        h = _fp16_from_bits(s & 0xFFFF).float()
        # fp16 fma: single rounding of the exact product-sum
        return (h * _MUL1_K_INV + _MUL1_K_BIAS).half().float()
    if codebook == 1:  # mcg
        x = (win * 0xCBAC1FED) & 0xFFFFFFFF
    else:  # legacy 3inst
        x = (win * 89226354 + 64248484) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = _fp16_from_bits(x & 0xFFFF).float()
    hi = _fp16_from_bits((x >> 16) & 0xFFFF).float()
    return (lo + hi).half().float()  # fp16 add, single rounding


def _unpack_windows(trellis: torch.Tensor, k_bits: int) -> torch.Tensor:
    """trellis int16 (T, 16K) -> windows (T, 256) int64.

    window(e) = bits [e*K + K - 16, e*K + K) of the tile bitstream
    (tail-biting, MSB-first within each LE uint32 word).
    """
    t = trellis.shape[0]
    w32 = trellis.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF  # (T, 8K)
    e = torch.arange(256, device=trellis.device, dtype=torch.int64)
    b0 = e * k_bits + k_bits - 16 + 256 * k_bits
    b1 = b0 + 16
    # NB: i0/i1 stay UNWRAPPED (biased by +256K bits) for the shift math;
    # the tail-biting modulo applies only to the array indexing, as in dq().
    i0u = b0 // 32
    i1u = (b1 - 1) // 32
    s0 = (i1u + 1) * 32 - b1
    a = w32[:, i0u % (8 * k_bits)]
    b = w32[:, i1u % (8 * k_bits)]
    return (((a << 32) | b) >> s0.unsqueeze(0)) & 0xFFFF


def dequant_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    codebook: int,
) -> torch.Tensor:
    """Reconstruct EXL3 linear layer(s) as fp32 W.T -> [..., out, in] (HF orientation).

    trellis: int16 (..., in/16, out/16, 16*K) -- arbitrary leading batch dims OK;
    suh fp16 (..., in); svh fp16 (..., out).
    """
    if trellis.dtype != torch.int16 or trellis.dim() < 3:
        raise ValueError(f"trellis must be int16 (..., in/16, out/16, 16K), got {trellis.shape} {trellis.dtype}")
    tk, tn, w = trellis.shape[-3:]
    if w % 16 != 0:
        raise ValueError(f"trellis last dim {w} not a multiple of 16")
    k_bits = w // 16
    if not 1 <= k_bits <= 8:
        raise ValueError(f"K={k_bits} out of range 1..8")
    in_f, out_f = tk * 16, tn * 16
    if in_f % _HAD_K or out_f % _HAD_K:
        raise ValueError(f"padded dims ({in_f}, {out_f}) must be multiples of {_HAD_K}")

    device = trellis.device
    lead = trellis.shape[:-3]
    win = _unpack_windows(trellis.reshape(-1, w), k_bits)  # (N, 256)
    vals = _decode_windows(win, codebook)  # (N, 256) fp32
    perm = _tensor_core_perm(device)
    tiles = torch.empty_like(vals)
    tiles.scatter_(1, perm.unsqueeze(0).expand_as(vals), vals)
    w_hat = (
        tiles.reshape(*lead, tk, tn, 16, 16)
        .transpose(-3, -2)  # (..., tk, 16r, tn, 16c)
        .reshape(*lead, in_f, out_f)
    )  # (..., in, out)

    # block-diagonal hadamard un-rotation, both sides
    h = _hadamard128(device)
    w_hat = w_hat.reshape(*lead, in_f // _HAD_K, _HAD_K, out_f)
    w_hat = h @ w_hat  # left
    w_hat = w_hat.reshape(*lead, in_f, out_f // _HAD_K, _HAD_K)
    w_hat = w_hat @ h  # right

    w = (
        w_hat.reshape(*lead, in_f, out_f)
        * suh.float().unsqueeze(-1)
        * svh.float().unsqueeze(-2)
    )
    return w.mT.contiguous()  # [..., out, in], HF orientation
