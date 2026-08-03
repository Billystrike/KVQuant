import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache

    from models.cage_config import CageConfig
    from models.qwen3_cage import (
        Qwen3CageCache,
        install_qwen3_cage_attention,
        uninstall_qwen3_cage_attention,
    )

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


@unittest.skipUnless(QWEN3_AVAILABLE, "requires the frozen Transformers Qwen3 stack")
class Qwen3CageTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
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
        self.cage_config = CageConfig(
            cage_enable=True,
            cage_k_group_sizes=[2, 4, 8],
            cage_k_clip_percentiles=[1.0, 1.0, 1.0],
            cage_v_group_sizes=[2, 4, 8],
            cage_v_clip_percentiles=[1.0, 1.0, 1.0],
        )

    def _cache(self):
        return Qwen3CageCache(self.cage_config, residual_length=4)

    def test_installer_is_idempotent_and_reversible(self):
        self.assertEqual(install_qwen3_cage_attention(self.model), 2)
        self.assertEqual(install_qwen3_cage_attention(self.model), 0)
        self.assertEqual(uninstall_qwen3_cage_attention(self.model), 2)

    def test_prefill_matches_dynamic_cache_and_builds_gqa_assignments(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        install_qwen3_cage_attention(self.model)
        with torch.no_grad():
            reference = self.model(
                input_ids=input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
            cage_cache = self._cache()
            candidate = self.model(
                input_ids=input_ids,
                past_key_values=cage_cache,
                use_cache=True,
            )

        torch.testing.assert_close(candidate.logits, reference.logits)
        self.assertEqual(cage_cache.get_seq_length(), 5)
        self.assertEqual(cage_cache.key_quantized_lengths, [4, 4])
        self.assertEqual(cage_cache.value_quantized_lengths, [1, 1])
        self.assertEqual(len(cage_cache.layer_policies), 2)
        for policy in cage_cache.layer_policies:
            self.assertEqual([tuple(x.shape) for x in policy.key_bucket_indices], [(2, 2), (2, 3), (2, 3)])
            self.assertEqual([tuple(x.shape) for x in policy.value_bucket_indices], [(2, 2), (2, 3), (2, 3)])

    def test_fixed_uniform_ablation_preserves_bucket_shapes_and_is_strided(self):
        install_qwen3_cage_attention(self.model)
        config = CageConfig(
            cage_enable=True,
            cage_ablation=True,
            cage_k_importance="fixed_uniform",
            cage_v_importance="fixed_uniform",
            cage_k_group_sizes=[2, 4, 8],
            cage_k_clip_percentiles=[1.0, 1.0, 1.0],
            cage_v_group_sizes=[2, 4, 8],
            cage_v_clip_percentiles=[1.0, 1.0, 1.0],
        )
        cache = Qwen3CageCache(config, residual_length=4)
        with torch.no_grad():
            self.model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                past_key_values=cache,
                use_cache=True,
            )

        expected = (
            torch.tensor([[2, 5], [2, 5]]),
            torch.tensor([[0, 3, 6], [0, 3, 6]]),
            torch.tensor([[1, 4, 7], [1, 4, 7]]),
        )
        for policy in cache.layer_policies:
            for observed, target in zip(policy.key_bucket_indices, expected):
                self.assertTrue(torch.equal(observed.cpu(), target))
            for observed, target in zip(policy.value_bucket_indices, expected):
                self.assertTrue(torch.equal(observed.cpu(), target))

    def test_cache_continuation_flushes_without_requantizing_old_prefix(self):
        install_qwen3_cage_attention(self.model)
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
        self.assertTrue(all(torch.isfinite(x).all() for x in cache.key_cache))
        self.assertTrue(all(torch.isfinite(x).all() for x in cache.value_cache))

    def test_missing_query_context_and_unsupported_crop_fail_closed(self):
        cache = self._cache()
        states = torch.zeros(1, 2, 1, 8)
        with self.assertRaisesRegex(ValueError, "query_states"):
            cache.update(states, states, 0, {})

        install_qwen3_cage_attention(self.model)
        with torch.no_grad():
            self.model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                past_key_values=cache,
                use_cache=True,
            )
        with self.assertRaises(NotImplementedError):
            cache.crop(3)

    def test_multitoken_continuation_fails_before_mutating_seen_length(self):
        install_qwen3_cage_attention(self.model)
        cache = self._cache()
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


if __name__ == "__main__":
    unittest.main()
