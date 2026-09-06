"""EXL3 packed expert banks: on-the-fly dequant of the selected rows, then the
stock bf16 grouped GEMM.

Bank row layouts (uint8, per expert per layer; H = hidden, I = moe_intermediate,
tb = H*I*K/8 trellis bytes at this layer's K, tb_max = same at the checkpoint's
max K -- rows are padded to the max-K layout, scales at fixed offsets):

  gate_up: [gate.trellis (tb_max) | up.trellis (tb_max) | gate.suh (2H) | gate.svh (2I)
            | up.suh (2H) | up.svh (2I)]
  down:    [down.trellis (tb_max) | down.suh (2I) | down.svh (2H)]

Decode gathers the routed slot rows (fixed [bs*top_k] shapes, no host sync --
CUDA-graph safe); prefill loops over the layer's experts in blocks (bounded
temp) with out-of-block routes masked to zero weight.
"""

from __future__ import annotations

import torch

from freetoken.models.exl3 import dequant_exl3

# experts per prefill block; w1+w2 bf16 for a block ~= G x 8 MB at H=2048, I=512
# (plus the fla/attention workspace, on GPUs with ~0.1 GiB of VRAM headroom)
_PREFILL_BLOCK = 4


def _row_params(numel: int, H: int, I: int, kind: str, k_bits: int) -> tuple[int, int]:
    """(trellis bytes tb at this layer's K, field width tb_max at max K)."""
    tb = H * I * k_bits // 8
    if kind == "gate_up":
        tb_max = (numel - 4 * H - 4 * I) // 2
    else:
        tb_max = numel - 2 * H - 2 * I
    if tb <= 0 or tb > tb_max:
        raise ValueError(f"exl3 {kind} row of {numel} B does not fit H={H} I={I} K={k_bits}")
    return tb, tb_max


def _dequant_rows(
    rows: torch.Tensor, H: int, I: int, kind: str, cb: int, k_bits: int
) -> torch.Tensor:
    """[N, row_bytes] uint8 -> bf16 [N, 2I, H] (gate_up) or [N, H, I] (down).

    Rows are padded to the max-K field width; only the first tb bytes of each
    trellis field are valid for this layer's K. Dequant runs one expert at a
    time (peak fp32 intermediate = one matrix, ~8 MB) into a bf16 output --
    batched decode would need the whole block in fp32 at once.
    """
    tb, tb_max = _row_params(rows.shape[1], H, I, kind, k_bits)
    n = rows.shape[0]
    out = (
        torch.empty(n, 2 * I, H, dtype=torch.bfloat16, device=rows.device)
        if kind == "gate_up"
        else torch.empty(n, H, I, dtype=torch.bfloat16, device=rows.device)
    )

    def _w(trellis_bytes: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
           t_in: int, t_out: int) -> torch.Tensor:
        t = (
            trellis_bytes.contiguous()
            .view(torch.int16)
            .reshape(-1, t_in // 16, t_out // 16, 16 * k_bits)
        )
        return dequant_exl3(
            t,
            suh.contiguous().view(torch.float16),
            svh.contiguous().view(torch.float16),
            cb,
        )

    for j in range(n):
        row = rows[j : j + 1]
        if kind == "gate_up":
            gate = _w(row[:, :tb], row[:, 2 * tb_max : 2 * tb_max + 2 * H],
                      row[:, 2 * tb_max + 2 * H : 2 * tb_max + 2 * H + 2 * I], H, I)
            base = 2 * tb_max + 2 * H + 2 * I
            up = _w(row[:, tb_max : tb_max + tb], row[:, base : base + 2 * H],
                    row[:, base + 2 * H : base + 2 * H + 2 * I], H, I)
            out[j] = torch.cat([gate[0], up[0]], dim=0).to(torch.bfloat16)
        else:
            down = _w(row[:, :tb], row[:, tb_max : tb_max + 2 * I],
                      row[:, tb_max + 2 * I :], I, H)
            out[j] = down[0].to(torch.bfloat16)
    return out


def fused_experts_exl3(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    hidden_size: int,
    intermediate_size: int,
    codebook: int,
    k_bits: int,
    *,
    is_prefill: bool,
) -> torch.Tensor:
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    H, I = hidden_size, intermediate_size
    if not is_prefill:
        # decode: fixed [bs*top_k] row gather, identity remap (CUDA-graph safe)
        ids = topk_ids.flatten()
        w1 = _dequant_rows(gate_up[ids], H, I, "gate_up", codebook, k_bits)
        w2 = _dequant_rows(down[ids], H, I, "down", codebook, k_bits)
        remap = torch.arange(ids.numel(), device=ids.device, dtype=torch.int32).view_as(
            topk_ids
        )
        return fused_experts_decode_impl(
            hidden_states, w1, w2, topk_weights, remap, activation,
            apply_router_weight_on_input,
        )

    # prefill: expert blocks, out-of-block routes masked to zero weight
    num_experts = gate_up.shape[0]
    sel = topk_ids.unique()
    out = None
    for i in range(0, sel.numel(), _PREFILL_BLOCK):
        blk = sel[i : i + _PREFILL_BLOCK]
        g = blk.numel()
        w1 = _dequant_rows(gate_up[blk], H, I, "gate_up", codebook, k_bits)
        w2 = _dequant_rows(down[blk], H, I, "down", codebook, k_bits)
        mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=blk.device)
        mapping[blk] = torch.arange(g, dtype=torch.int32, device=blk.device)
        ids_l = mapping[topk_ids]
        mask = ids_l >= 0
        ids_l = ids_l.clamp_min(0).contiguous()
        w_l = torch.where(mask, topk_weights, topk_weights.new_zeros(())).contiguous()
        out_b = fused_experts_impl(
            hidden_states.clone(), w1, w2, w_l, ids_l, activation,
            apply_router_weight_on_input,
        )
        out = out_b if out is None else out + out_b
    return out
