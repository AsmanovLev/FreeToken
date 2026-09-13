from __future__ import annotations

from types import SimpleNamespace

from freetoken.models.qwen3_5_moe.config import _ct_covers, _ct_targets_and_ignore, parse_config


def _hf(quantization_config):
    return SimpleNamespace(
        quantization_config=quantization_config,
        text_config=SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            vocab_size=248320,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            num_hidden_layers=40,
            max_position_embeddings=262144,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            tie_word_embeddings=False,
            intermediate_size=0,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        ),
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )


def test_ct_covers_experts_only():
    hf_config = _hf({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                            "group_size": 16, "symmetric": True, "dynamic": False},
                "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                      "group_size": 16, "symmetric": True, "dynamic": "local"},
                "targets": [
                    "re:.*\\.layers\\.(0|1|2|...|39)\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
                ],
            }
        },
        "ignore": ["lm_head", "re:.*embed_tokens.*", "re:.*visual.*", "re:.*norm.*",
                   "re:.*self_attn.*", "re:.*linear_attn.*", "re:.*\\.mlp\\.gate$",
                   "re:.*shared_expert.*"],
    })
    assert _ct_covers(hf_config, "mlp.experts."), "experts regex should match mlp.experts."
    assert not _ct_covers(hf_config, "self_attn."), "self_attn not in targets"
    assert not _ct_covers(hf_config, "linear_attn."), "linear_attn not in targets"
    assert not _ct_covers(hf_config, "shared_expert."), "shared_expert in ignore"
    assert not _ct_covers(hf_config, ".mlp.gate_proj"), "dense mlp gate not targeted"


def test_ct_covers_bare_linear():
    hf_config = _hf({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                            "group_size": 16, "symmetric": True, "dynamic": False},
                "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                      "group_size": 16, "symmetric": True, "dynamic": "local"},
                "targets": ["Linear"],
            }
        },
        "ignore": ["re:.*norm.*", "lm_head"],
    })
    assert _ct_covers(hf_config, "mlp.experts."), "bare Linear covers experts"
    assert _ct_covers(hf_config, "self_attn."), "bare Linear covers self_attn"
    assert _ct_covers(hf_config, "shared_expert."), "bare Linear covers shared_expert"


def test_parse_config_ct_experts_only():
    hf_config = _hf({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                            "group_size": 16, "symmetric": True, "dynamic": False},
                "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                      "group_size": 16, "symmetric": True, "dynamic": "local"},
                "targets": [
                    "re:.*\\.layers\\.(0|1|2|...|39)\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
                ],
            }
        },
        "ignore": ["lm_head", "re:.*embed_tokens.*", "re:.*visual.*", "re:.*norm.*",
                   "re:.*self_attn.*", "re:.*linear_attn.*", "re:.*\\.mlp\\.gate$",
                   "re:.*shared_expert.*"],
    })
    cfg = parse_config(hf_config)
    assert cfg.expert_quant == "nvfp4", f"expected nvfp4, got {cfg.expert_quant}"
    assert cfg.attn_quant == "none", f"expected none, got {cfg.attn_quant}"
    assert cfg.dense_quant == "none", f"expected none, got {cfg.dense_quant}"
    assert cfg.lm_head_quant == "none", f"expected none, got {cfg.lm_head_quant}"


def test_parse_config_ct_bare_linear():
    hf_config = _hf({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                            "group_size": 16, "symmetric": True, "dynamic": False},
                "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                      "group_size": 16, "symmetric": True, "dynamic": "local"},
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head", "re:.*norm.*"],
    })
    cfg = parse_config(hf_config)
    assert cfg.expert_quant == "nvfp4", f"expected nvfp4, got {cfg.expert_quant}"
    assert cfg.attn_quant == "nvfp4", f"expected nvfp4, got {cfg.attn_quant}"
    assert cfg.dense_quant == "nvfp4", f"expected nvfp4, got {cfg.dense_quant}"


def test_parse_config_modelopt_nvfp4_unchanged():
    hf_config = _hf({
        "quant_algo": "NVFP4",
        "kv_cache_dtype": "nf4",
    })
    cfg = parse_config(hf_config)
    assert cfg.expert_quant == "nvfp4", f"expected nvfp4, got {cfg.expert_quant}"


def test_parse_config_modelopt_mixed_fp8_nvfp4():
    hf_config = _hf({
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
            "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        },
    })
    cfg = parse_config(hf_config)
    assert cfg.expert_quant == "nvfp4", f"expected nvfp4, got {cfg.expert_quant}"
    assert cfg.attn_quant == "fp8_pertensor", f"expected fp8_pertensor, got {cfg.attn_quant}"
