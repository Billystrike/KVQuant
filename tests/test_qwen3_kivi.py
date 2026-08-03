import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache

    from models.qwen3_kivi import Qwen3KiviCache, Qwen3KiviCacheConfig

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


@unittest.skipUnless(QWEN3_AVAILABLE, "requires the frozen Transformers Qwen3 stack")
class Qwen3KiviTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            attention_dropout=0.0,
            use_cache=True,
        )
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()

    def _cache(self):
        return Qwen3KiviCache(Qwen3KiviCacheConfig(group_size=4, residual_length=4))

    def test_config_rejects_non_grid_residual(self):
        with self.assertRaisesRegex(ValueError, "multiple of group_size"):
            Qwen3KiviCacheConfig(group_size=4, residual_length=6)

    def test_prefill_matches_dynamic_cache_and_quantizes_persistent_storage(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            reference_cache = DynamicCache()
            reference = self.model(
                input_ids=input_ids,
                past_key_values=reference_cache,
                use_cache=True,
            )
            kivi_cache = self._cache()
            candidate = self.model(
                input_ids=input_ids,
                past_key_values=kivi_cache,
                use_cache=True,
            )

        torch.testing.assert_close(candidate.logits, reference.logits)
        self.assertEqual(kivi_cache.get_seq_length(), 5)
        self.assertEqual(kivi_cache.key_quantized_lengths, [4, 4])
        self.assertEqual(kivi_cache.value_quantized_lengths, [1, 1])
        self.assertTrue(
            any(
                not torch.equal(kivi_cache.key_cache[i], reference_cache.key_cache[i])
                for i in range(2)
            )
        )
        self.assertFalse(hasattr(kivi_cache, "key_bucket_indices"))

    def test_continuation_flushes_and_preserves_old_quantized_prefix(self):
        cache = self._cache()
        with torch.no_grad():
            first = self.model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
                past_key_values=cache,
                use_cache=True,
            )
            old_key_prefix = cache.key_cache[0][..., :4, :].clone()
            second = self.model(
                input_ids=torch.tensor([[8]]),
                past_key_values=first.past_key_values,
                use_cache=True,
            )

        self.assertIs(second.past_key_values, cache)
        self.assertEqual(cache.get_seq_length(), 8)
        self.assertEqual(cache.key_quantized_lengths, [8, 8])
        self.assertEqual(cache.value_quantized_lengths, [4, 4])
        self.assertTrue(torch.equal(cache.key_cache[0][..., :4, :], old_key_prefix))
        self.assertTrue(torch.isfinite(second.logits).all())

    def test_unsupported_batch_chunk_and_crop_fail_closed(self):
        cache = self._cache()
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            cache.update(torch.zeros(2, 2, 1, 8), torch.zeros(2, 2, 1, 8), 0)

        with torch.no_grad():
            self.model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                past_key_values=cache,
                use_cache=True,
            )
            with self.assertRaisesRegex(ValueError, "one token per update"):
                self.model(
                    input_ids=torch.tensor([[6, 7]]),
                    past_key_values=cache,
                    use_cache=True,
                )
        self.assertEqual(cache.get_seq_length(), 5)
        with self.assertRaises(NotImplementedError):
            cache.crop(3)


if __name__ == "__main__":
    unittest.main()
