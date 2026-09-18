"""ExLlamaV3 (EXL3) checkpoints.

``quantization_config.quant_method == "exl3"`` quantizes only the routed MoE
experts (QTIP trellis tables); the dense weights -- attention, GDN, shared
expert, lm_head -- stay bf16/fp16. Expert packing is owned by the family's
``setup_offload_expert_banks`` hook, so as far as QuantConfig is concerned every
module is unquantized.
"""

from __future__ import annotations

from typing import Any

from ..registry import LayerKind, register_dialect
from ..scheme import QuantScheme
from .base import QuantConfig


@register_dialect
class Exl3Config(QuantConfig):
    """EXL3 experts are served from the offload cache, not a QuantMethod."""

    dialect = "exl3"

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return None

    def get_quant_method(self, layer: Any, prefix: str):
        """A bf16 method for dense Linear, but NO method for the MoE layer.

        The routed experts are packed by the family's ``setup_offload_expert_banks``
        hook and served through the ``quant_format == "exl3"`` branch of
        ``OffloadMoELayer._expert_gemm``; handing the MoE layer an
        ``UnquantizedMoEMethod`` would make the engine demand bf16 banks and
        reject the packed trellis rows.
        """
        if layer.quant_layer_kind == LayerKind.MOE:
            return None
        return super().get_quant_method(layer, prefix)
