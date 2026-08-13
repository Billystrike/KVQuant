import unittest

import torch

from models.cage_importance import compute_key_importance
from models.qwen3_cage_v4 import (
    CageV4DTQIConfig,
    Qwen3CageV4DTQICache,
    compute_dtqi_key_importance,
)


class CageV4DTQITest(unittest.TestCase):
    def test_recent_window_equal_to_sequence_reduces_to_original_importance(self):
        torch.manual_seed(1)
        query = torch.randn(2, 4, 6, 3)
        key = torch.randn(2, 2, 6, 3)
        expected = compute_key_importance(query, key, num_key_value_groups=2)
        observed = compute_dtqi_key_importance(
            query,
            key,
            num_key_value_groups=2,
            recent_query_window=6,
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_recent_query_energy_can_change_channel_ranking(self):
        query = torch.ones(1, 1, 4, 2)
        query[:, :, :2, 0] = 10.0
        query[:, :, -2:, 1] = 6.0
        key = torch.tensor([[[[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [2.0, 2.0]]]])
        global_only = compute_key_importance(query, key, num_key_value_groups=1)
        dtqi = compute_dtqi_key_importance(
            query,
            key,
            num_key_value_groups=1,
            recent_query_window=2,
        )
        self.assertEqual(int(global_only.argmax(dim=-1).item()), 0)
        self.assertEqual(int(dtqi.argmax(dim=-1).item()), 1)

    def test_cache_binds_recent_window_to_residual_and_keeps_v3_storage_config(self):
        config = CageV4DTQIConfig(
            one_bit_channels=[0],
            two_bit_channels=[2],
            key_base_group_size=4,
            key_refinement_group_size=4,
            value_group_size=4,
            sink_length=1,
        )
        cache = Qwen3CageV4DTQICache(config, residual_length=4)
        query = torch.randn(1, 2, 7, 4)
        key = torch.randn(1, 1, 7, 4)
        value = torch.randn_like(key)
        policy = cache._build_layer_policy(
            query_states=query,
            key_states=key,
            value_states=value,
            o_proj_weight=torch.randn(8, 8),
            layer_idx=0,
        )
        self.assertEqual(policy.one_bit_channels, 0)
        self.assertEqual(policy.two_bit_channels, 2)
        self.assertEqual(list(policy.one_bit_indices.shape), [1, 0])
        self.assertEqual(list(policy.two_bit_indices.shape), [1, 2])
        self.assertFalse(hasattr(policy, "value_bucket_indices"))
        self.assertEqual(cache.residual_length, 4)

    def test_invalid_shapes_and_windows_fail_closed(self):
        query = torch.ones(1, 1, 4, 2)
        key = torch.ones(1, 1, 4, 2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compute_dtqi_key_importance(
                query,
                key,
                num_key_value_groups=1,
                recent_query_window=0,
            )
        with self.assertRaisesRegex(ValueError, "head counts"):
            compute_dtqi_key_importance(
                query,
                key,
                num_key_value_groups=2,
                recent_query_window=2,
            )


if __name__ == "__main__":
    unittest.main()
