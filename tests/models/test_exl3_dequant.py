"""EXL3 dequant tests (models/exl3.py).

Golden vectors in exl3_golden.npz were produced by exllamav3's own CUDA
reconstruct kernel (exllamav3_ext.reconstruct) on synthetic trellis data for
all K=1..8 x codebooks, plus one real layer (gate_proj, K=3, mcg) from
0xSero/Ornith-1.5-35B-A3B-EXL3-2.75bpw with the hadamard/scale epilogue.
"""

import pathlib

import numpy as np
import pytest
import torch

from freetoken.models.exl3 import dequant_exl3, _decode_windows, _tensor_core_perm, _unpack_windows

GOLDEN = pathlib.Path(__file__).with_name("exl3_golden.npz")


def _my_what(trellis: torch.Tensor, codebook: int) -> torch.Tensor:
    tk, tn, w = trellis.shape
    k = w // 16
    win = _unpack_windows(trellis.reshape(tk * tn, w), k)
    vals = _decode_windows(win, codebook)
    perm = _tensor_core_perm(trellis.device)
    tiles = torch.empty_like(vals)
    tiles.scatter_(1, perm.unsqueeze(0).expand_as(vals), vals)
    return tiles.reshape(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


@pytest.mark.parametrize("k", range(1, 9))
def test_reconstruct_mcg(k):
    d = np.load(GOLDEN)
    trellis = torch.from_numpy(d[f"synth_K{k}_mcg.trellis"])
    want = torch.from_numpy(d[f"synth_K{k}_mcg.what"])
    got = _my_what(trellis, codebook=1)
    assert torch.equal(got.half(), want), f"K={k} mcg mismatch"


@pytest.mark.parametrize("cb,name", [(0, "legacy"), (2, "mul1")])
def test_reconstruct_codebooks(cb, name):
    d = np.load(GOLDEN)
    trellis = torch.from_numpy(d[f"synth_K3_{name}.trellis"])
    want = torch.from_numpy(d[f"synth_K3_{name}.what"])
    got = _my_what(trellis, codebook=cb)
    assert torch.equal(got.half(), want), f"{name} mismatch"


def test_real_layer_full_pipeline():
    """gate_proj (K=3, mcg) from the real EXL3 checkpoint: windows+decode+perm
    bit-exact vs the CUDA oracle's W_hat; full pipeline (hadamard+scales) matches
    up to fp16 storage rounding of the golden."""
    d = np.load(GOLDEN)
    trellis = torch.from_numpy(d["real_gate.trellis"])   # (128, 32, 48)
    suh = torch.from_numpy(d["real_gate.suh"]).half()
    svh = torch.from_numpy(d["real_gate.svh"]).half()
    want_what = torch.from_numpy(d["real_gate.what"])    # (in, out) fp16
    want_w = torch.from_numpy(d["real_gate.w"])          # (in, out) fp16

    got_what = _my_what(trellis, codebook=1)
    assert torch.equal(got_what.half(), want_what)

    got_w = dequant_exl3(trellis, suh, svh, codebook=1).t()  # -> (in, out)
    got_w = got_w.float()
    want_wf = want_w.float()
    cos = torch.nn.functional.cosine_similarity(got_w.flatten(), want_wf.flatten(), dim=0)
    rel = (got_w - want_wf).abs().max() / want_wf.abs().max()
    assert cos > 0.999999 and rel < 1e-2, f"full pipeline cos={cos} rel={rel}"


def test_input_validation():
    with pytest.raises(ValueError):
        dequant_exl3(torch.zeros(2, 2, 10, dtype=torch.int16),
                     torch.zeros(32, dtype=torch.float16),
                     torch.zeros(32, dtype=torch.float16), 1)
