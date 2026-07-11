import json
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path

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

from models.cage_cache import unpack_cage_past_key_value
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

    def test_cage_prefill_attention_is_fp16_and_cache_uses_kivi_buffers(self):
        torch.manual_seed(0)
        quantized = self._attention()
        reference = self._attention()
        reference.load_state_dict(quantized.state_dict())
        reference.cage_config.cage_k_enable = False
        reference.cage_config.cage_v_enable = False
        hidden = torch.randn(1, 5, 16)
        positions = torch.arange(5).unsqueeze(0)
        mask = torch.zeros(1, 1, 5, 5)

        actual, _, cache = quantized(hidden, attention_mask=mask, position_ids=positions, use_cache=True)
        expected, _, _ = reference(hidden, attention_mask=mask, position_ids=positions, use_cache=False)

        torch.testing.assert_close(actual, expected)
        unpacked = unpack_cage_past_key_value(cache)
        self.assertEqual(unpacked.key_cache.key_quant_buckets[0].shape[-2], 4)
        self.assertEqual(unpacked.key_cache.key_full.shape[-2], 1)
        self.assertEqual(unpacked.value_cache.value_quant_buckets[0].shape[-2], 3)
        self.assertEqual(unpacked.value_cache.value_full.shape[-2], 2)

    def test_cage_prefill_collects_and_dumps_perturbation_metrics_when_enabled(self):
        torch.manual_seed(0)
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.cage_collect_metrics = True
            config.cage_dump_dir = tmpdir
            attention = LlamaFlashAttention_KIVI(config)
            attention.eval()

            hidden_states = torch.randn(1, 3, 16)
            position_ids = torch.arange(3).unsqueeze(0)

            attention(
                hidden_states,
                attention_mask=torch.zeros(1, 1, 3, 3),
                position_ids=position_ids,
                use_cache=True,
            )

            self.assertIsInstance(attention.last_cage_metrics, dict)
            output_path = Path(tmpdir) / "cage_perturbation_metrics.jsonl"
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(record["phase"], "prefill")
            self.assertEqual(record["attention_module"], "LlamaFlashAttention_KIVI")
            self.assertIn("attention_logit_mse", record)

    def test_cage_decode_flushes_key_block_and_rolls_value_buffer(self):
        attention = self._attention()
        prefill = torch.randn(1, 5, 16)
        _, _, cache = attention(
            prefill,
            attention_mask=torch.zeros(1, 1, 5, 5),
            position_ids=torch.arange(5).unsqueeze(0),
            use_cache=True,
        )
        _, _, updated = attention(
            torch.randn(1, 1, 16),
            attention_mask=torch.zeros(1, 1, 1, 6),
            position_ids=torch.tensor([[5]]),
            past_key_value=cache,
            use_cache=True,
        )
        unpacked = unpack_cage_past_key_value(updated)
        self.assertIsNone(unpacked.key_cache.key_full)
        self.assertEqual(unpacked.key_cache.key_quant_buckets[0].shape[-2], 6)
        self.assertEqual(unpacked.value_cache.value_quant_buckets[0].shape[-2], 4)
        self.assertEqual(unpacked.value_cache.value_full.shape[-2], 2)


if __name__ == "__main__":
    unittest.main()
