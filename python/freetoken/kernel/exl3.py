"""Fused EXL3 dequant CUDA kernel (csrc/jit/exl3_dequant.cu): packed rows -> bf16.

One launch dequantizes M same-shape matrices; used by moe/fused_exl3.py to
replace the torch-level reference dequant (dozens of small launches per
matrix) in the hot MoE path.
"""

from __future__ import annotations

import functools

import torch

from .utils import load_jit, make_cpp_args


@functools.cache
def _exl3_dequant_module(k_bits: int):
    cpp_args = make_cpp_args(k_bits, 128)  # <k_bits, num_threads>
    return load_jit(
        "exl3_dequant",
        *cpp_args,
        cuda_files=["exl3_dequant.cu"],
        cuda_wrappers=[("run", f"Exl3DequantKernel<{cpp_args}>::run")],
    )


def exl3_dequant_fused(
    src: torch.Tensor,
    out: torch.Tensor,
    trellis_off: int,
    suh_off: int,
    svh_off: int,
    in_dim: int,
    out_dim: int,
    codebook: int,
    k_bits: int,
) -> None:
    """Dequantize M packed EXL3 matrices in ``src`` (M, row_bytes uint8) into
    ``out`` (M, out_dim, in_dim) bf16, HF orientation.

    Byte offsets point at the trellis/suh/svh fields within each row.
    """
    if not 1 <= k_bits <= 8:
        raise ValueError(f"k_bits {k_bits} out of range 1..8")
    _exl3_dequant_module(k_bits).run(
        src, out, trellis_off, suh_off, svh_off, in_dim, out_dim, codebook
    )
