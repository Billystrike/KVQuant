from __future__ import annotations

import json
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path

try:
    import torch
    from transformers.generation import GenerationMixin  # noqa: F401
    from transformers.models.llama.configuration_llama import LlamaConfig

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _install_optional_cuda_stubs():
    triton = types.ModuleType("triton")
    triton.__spec__ = ModuleSpec("triton", loader=None)
    triton.jit = lambda fn=None, **_: fn if fn is not None else lambda wrapped: wrapped
    triton.cdiv = lambda x, y: (x + y - 1) // y
    language = types.ModuleType("triton.language")
    language.__spec__ = ModuleSpec("triton.language", loader=None)
    language.constexpr = object()
    triton.language = language
    gemv = types.ModuleType("kivi_gemv")
    gemv.__spec__ = ModuleSpec("kivi_gemv", loader=None)
    gemv.gemv_forward_cuda_outer_dim = lambda *_, **__: None
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", language)
    sys.modules.setdefault("kivi_gemv", gemv)


if TORCH_AVAILABLE:
    _install_optional_cuda_stubs()

    from models.llama_cage_v3 import (
        LlamaCageV3Config,
        LlamaCageV3Plan,
        append_decode_token,
        build_prefill_cache,
        config_from_quota_payload,
        install_llama_cage_v3_config,
        reconstruct_key,
        reconstruct_value,
        unpack_cache,
    )
    from models.llama_kivi import LlamaFlashAttention_KIVI


REPO_ROOT = Path(__file__).resolve().parents[1]
QUOTA_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quota_plan_v1.json"


def _tiny_policy() -> LlamaCageV3Config:
    return LlamaCageV3Config(
        plans=(
            LlamaCageV3Plan(
                prompt_length=7,
                residual_length=4,
                sink_length=1,
                layer_two_bit_channel_quotas=(2,),
            ),
        ),
        key_base_group_size=4,
        key_refinement_group_size=4,
        value_group_size=4,
    )


def _tiny_model_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=64,
        vocab_size=64,
        attention_dropout=0.0,
    )
    config.use_flash = True
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = 4
    config.residual_length = 4
    config.cage_enable = False
    config.cage_v3_enable = True
    config.cage_v3_plans = [
        {
            "prompt_length": 7,
            "residual_length": 4,
            "sink_length": 1,
            "layer_two_bit_channel_quotas": [2],
        }
    ]
    config.cage_v3_key_base_group_size = 4
    config.cage_v3_key_refinement_group_size = 4
    config.cage_v3_value_group_size = 4
    return config


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch and Transformers")
class LlamaCageV3RuntimeTest(unittest.TestCase):
    def test_checked_in_quota_installs_only_on_exact_production_architecture(self):
        payload = json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
        policy = config_from_quota_payload(payload)
        self.assertEqual(policy.num_hidden_layers, 32)
        self.assertEqual([plan.prompt_length for plan in policy.plans], [1024, 2048, 4032])
        config = types.SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            cage_enable=True,
        )
        installed = install_llama_cage_v3_config(config, payload)
        self.assertEqual(installed, policy)
        self.assertTrue(config.cage_v3_enable)
        self.assertFalse(config.cage_enable)
        config.num_hidden_layers = 31
        with self.assertRaisesRegex(ValueError, "architecture"):
            install_llama_cage_v3_config(config, payload)

    def test_prefill_preserves_sink_and_builds_sparse_key_uniform_value_cache(self):
        torch.manual_seed(31)
        query = torch.randn(1, 4, 7, 8)
        key = torch.randn(1, 2, 7, 8)
        value = torch.randn(1, 2, 7, 8)
        cache = build_prefill_cache(
            _tiny_policy(),
            query_states=query,
            key_states=key,
            value_states=value,
            layer_idx=0,
        )
        self.assertEqual(tuple(cache.two_bit_indices.shape), (2, 2))
        torch.testing.assert_close(cache.key_sink, key[:, :, :1, :])
        torch.testing.assert_close(cache.value_sink, value[:, :, :1, :])
        self.assertEqual(cache.key_quantized.shape[-2], 4)
        self.assertEqual(cache.key_residual.shape[-2], 2)
        self.assertEqual(cache.value_quantized.shape[-2], 2)
        self.assertEqual(cache.value_residual.shape[-2], 4)
        self.assertEqual(reconstruct_key(cache).shape[-2], 7)
        self.assertEqual(reconstruct_value(cache).shape[-2], 7)
        self.assertFalse(torch.equal(cache.key_quantized, key[:, :, 1:5, :]))
        self.assertFalse(torch.equal(cache.value_quantized, value[:, :, 1:3, :]))

    def test_continuation_flushes_key_and_rolls_value_without_requantizing_old_prefix(self):
        torch.manual_seed(32)
        policy = _tiny_policy()
        cache = build_prefill_cache(
            policy,
            query_states=torch.randn(1, 4, 7, 8),
            key_states=torch.randn(1, 2, 7, 8),
            value_states=torch.randn(1, 2, 7, 8),
            layer_idx=0,
        )
        old_key = cache.key_quantized.clone()
        old_value = cache.value_quantized.clone()
        for _ in range(2):
            cache = append_decode_token(
                policy,
                cache,
                key_states=torch.randn(1, 2, 1, 8),
                value_states=torch.randn(1, 2, 1, 8),
            )
        self.assertIsNone(cache.key_residual)
        self.assertEqual(cache.key_quantized.shape[-2], 8)
        self.assertEqual(cache.value_quantized.shape[-2], 4)
        self.assertEqual(cache.value_residual.shape[-2], 4)
        torch.testing.assert_close(cache.key_quantized[:, :, :4, :], old_key)
        torch.testing.assert_close(cache.value_quantized[:, :, :2, :], old_value)
        self.assertEqual(cache.kv_seq_len, 9)

    def test_attention_prefill_is_exact_and_decode_uses_v3_cache(self):
        torch.manual_seed(33)
        attention = LlamaFlashAttention_KIVI(_tiny_model_config(), layer_idx=0).eval()
        hidden = torch.randn(1, 7, 32)
        positions = torch.arange(7).unsqueeze(0)
        mask = torch.zeros(1, 1, 7, 7)
        with torch.no_grad():
            exact, _, _ = attention(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            )
            actual, _, packed = attention(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
            )
        torch.testing.assert_close(actual, exact)
        cache = unpack_cache(packed)
        self.assertEqual(cache.kv_seq_len, 7)
        with torch.no_grad():
            output, _, updated = attention(
                torch.randn(1, 1, 32),
                attention_mask=torch.zeros(1, 1, 1, 8),
                position_ids=torch.tensor([[7]]),
                past_key_value=packed,
                use_cache=True,
            )
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(unpack_cache(updated).kv_seq_len, 8)

    def test_unsupported_prompt_batch_and_multitoken_continuation_fail_closed(self):
        policy = _tiny_policy()
        with self.assertRaisesRegex(ValueError, "frozen.*prompt|plan for prompt"):
            build_prefill_cache(
                policy,
                query_states=torch.randn(1, 4, 6, 8),
                key_states=torch.randn(1, 2, 6, 8),
                value_states=torch.randn(1, 2, 6, 8),
                layer_idx=0,
            )
        with self.assertRaisesRegex(ValueError, "batch size one"):
            build_prefill_cache(
                policy,
                query_states=torch.randn(2, 4, 7, 8),
                key_states=torch.randn(2, 2, 7, 8),
                value_states=torch.randn(2, 2, 7, 8),
                layer_idx=0,
            )
        cache = build_prefill_cache(
            policy,
            query_states=torch.randn(1, 4, 7, 8),
            key_states=torch.randn(1, 2, 7, 8),
            value_states=torch.randn(1, 2, 7, 8),
            layer_idx=0,
        )
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            append_decode_token(
                policy,
                cache,
                key_states=torch.randn(1, 2, 2, 8),
                value_states=torch.randn(1, 2, 2, 8),
            )


if __name__ == "__main__":
    unittest.main()
