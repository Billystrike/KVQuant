import sys
import types
import unittest
from importlib.machinery import ModuleSpec

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

# Force Transformers to resolve GenerationMixin before the local Triton stub is
# installed; recent Transformers versions use lazy imports that inspect Triton.
from transformers.generation import GenerationMixin  # noqa: F401


def _install_optional_cuda_stubs():
    triton = types.ModuleType("triton")
    triton.__spec__ = ModuleSpec("triton", loader=None)
    triton.jit = lambda fn=None, **_: fn if fn is not None else lambda wrapped: wrapped
    triton.cdiv = lambda x, y: (x + y - 1) // y

    triton_language = types.ModuleType("triton.language")
    triton_language.__spec__ = ModuleSpec("triton.language", loader=None)
    triton_language.constexpr = object()
    triton.language = triton_language

    kivi_gemv = types.ModuleType("kivi_gemv")
    kivi_gemv.__spec__ = ModuleSpec("kivi_gemv", loader=None)
    kivi_gemv.gemv_forward_cuda_outer_dim = lambda *_, **__: None

    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", triton_language)
    sys.modules.setdefault("kivi_gemv", kivi_gemv)


_install_optional_cuda_stubs()

from models.cage_cache import is_cage_past_key_value, unpack_cage_past_key_value
from models.llama_kivi import LlamaFlashAttention_KIVI


class CageAttentionIntegrationTest(unittest.TestCase):
    def _config(self):
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=64,
            vocab_size=32,
        )
        config.use_flash = True
        config.k_bits = 2
        config.v_bits = 2
        config.group_size = 2
        config.residual_length = 2
        config.cage_enable = True
        config.cage_mode = "fake"
        config.cage_k_group_sizes = [2, 2, 2]
        config.cage_k_clip_percentiles = [1.0, 1.0, 1.0]
        config.cage_k_num_buckets = 3
        config.cage_v_group_sizes = [2, 2, 2]
        config.cage_v_clip_percentiles = [1.0, 1.0, 1.0]
        config.cage_v_num_buckets = 3
        return config

    def _attention(self):
        attention = LlamaFlashAttention_KIVI(self._config())
        attention.eval()

        def _fail_if_original_flash_path_is_used(*_, **__):
            raise AssertionError("original flash path should not be used when CAGE fake mode is enabled")

        attention._flash_attention_forward = _fail_if_original_flash_path_is_used
        return attention

    def test_cage_prefill_uses_fake_path_and_returns_cage_cache(self):
        torch.manual_seed(0)
        attention = self._attention()
        hidden_states = torch.randn(1, 3, 16)
        position_ids = torch.arange(3).unsqueeze(0)
        attention_mask = torch.zeros(1, 1, 3, 3)

        output, attn_weights, past_key_value = attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        self.assertEqual(output.shape, (1, 3, 16))
        self.assertIsNone(attn_weights)
        self.assertTrue(is_cage_past_key_value(past_key_value))

        unpacked = unpack_cage_past_key_value(past_key_value)
        self.assertEqual(unpacked.kv_seq_len, 3)
        self.assertEqual(len(unpacked.key_cache.key_quant_buckets), 3)
        self.assertEqual(len(unpacked.value_cache.value_quant_buckets), 3)
        self.assertIsNone(unpacked.key_cache.key_full)
        self.assertIsNone(unpacked.value_cache.value_full)

    def test_cage_decode_reuses_bucket_policy_and_appends_full_precision_token(self):
        torch.manual_seed(0)
        attention = self._attention()
        prefill_hidden = torch.randn(1, 3, 16)
        prefill_positions = torch.arange(3).unsqueeze(0)

        _, _, prefill_cache = attention(
            prefill_hidden,
            attention_mask=torch.zeros(1, 1, 3, 3),
            position_ids=prefill_positions,
            use_cache=True,
        )
        prefill_unpacked = unpack_cage_past_key_value(prefill_cache)

        decode_hidden = torch.randn(1, 1, 16)
        output, _, decode_cache = attention(
            decode_hidden,
            attention_mask=torch.zeros(1, 1, 1, 4),
            position_ids=torch.tensor([[3]]),
            past_key_value=prefill_cache,
            use_cache=True,
        )

        self.assertEqual(output.shape, (1, 1, 16))
        decode_unpacked = unpack_cage_past_key_value(decode_cache)
        self.assertEqual(decode_unpacked.kv_seq_len, 4)
        self.assertEqual(decode_unpacked.key_cache.key_full.shape[-2], 1)
        self.assertEqual(decode_unpacked.value_cache.value_full.shape[-2], 1)
        for before, after in zip(
            prefill_unpacked.key_cache.key_bucket_indices,
            decode_unpacked.key_cache.key_bucket_indices,
        ):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(
            prefill_unpacked.value_cache.value_bucket_indices,
            decode_unpacked.value_cache.value_bucket_indices,
        ):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
