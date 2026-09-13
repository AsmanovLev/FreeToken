from __future__ import annotations

# Module-prefix skip flags (--skip-{vision,mtp,carry}) rebuild a module-level prefix set
# that _rename and the compressed-tensors dense iterator consult. These tests exercise the
# prefix set + _rename directly, no GPU/checkpoint needed.

_VISION = "model.visual.blocks.0.attn.qkv.weight"
_VISION_BARE = "visual.blocks.0.attn.qkv.weight"
_MTP = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
_CARRY = "model.carry.0.weight"
_TEXT = "model.language_model.layers.0.input_layernorm.weight"


def _import():
    from freetoken.models.qwen3_5_moe.weight import configure_skip_modules, _rename

    return configure_skip_modules, _rename


def test_defaults_skip_vision_mtp_and_carry():
    configure, rename = _import()
    try:
        configure(vision=True, mtp=True, carry=True)
        assert rename(_VISION) is None
        assert rename(_VISION_BARE) is None
        assert rename(_MTP) is None
        assert rename(_CARRY) is None
        assert rename(_TEXT) == "model.layers.0.input_layernorm.weight"
    finally:
        configure(vision=True, mtp=True, carry=True)


def test_skip_carry_off_loads_carry_only():
    configure, rename = _import()
    try:
        configure(vision=True, mtp=True, carry=False)
        assert rename(_CARRY) == "model.carry.0.weight"
        assert rename(_VISION) is None
        assert rename(_MTP) is None
    finally:
        configure(vision=True, mtp=True, carry=True)


def test_skip_vision_off_loads_both_visual_spellings():
    configure, rename = _import()
    try:
        configure(vision=False, mtp=True, carry=True)
        assert rename(_VISION) == "model.visual.blocks.0.attn.qkv.weight"
        assert rename(_VISION_BARE) == "visual.blocks.0.attn.qkv.weight"
        assert rename(_MTP) is None
        assert rename(_CARRY) is None
    finally:
        configure(vision=True, mtp=True, carry=True)


def test_skip_mtp_off_loads_mtp_only():
    configure, rename = _import()
    try:
        configure(vision=True, mtp=False, carry=True)
        assert rename(_MTP) == "mtp.layers.0.mlp.experts.0.gate_proj.weight"
        assert rename(_VISION) is None
        assert rename(_CARRY) is None
    finally:
        configure(vision=True, mtp=True, carry=True)


def test_everything_off_loads_all_modules():
    configure, rename = _import()
    try:
        configure(vision=False, mtp=False, carry=False)
        assert rename(_VISION) is not None
        assert rename(_MTP) is not None
        assert rename(_CARRY) is not None
        assert rename(_TEXT) is not None
    finally:
        configure(vision=True, mtp=True, carry=True)


def test_configure_is_idempotent():
    configure, rename = _import()
    try:
        configure(vision=False, mtp=True, carry=False)
        configure(vision=False, mtp=True, carry=False)
        assert rename(_VISION) is not None
        assert rename(_MTP) is None
        assert rename(_CARRY) is not None
    finally:
        configure(vision=True, mtp=True, carry=True)
