import json
import unittest

import torch

from models.cage_cache import CageKeyCache, CageValueCache, pack_cage_past_key_value
from utils.cage_memory import (
    estimate_cage_cache_bytes,
    estimate_kivi_cache_bytes,
    summarize_cache_bytes,
)


class CageMemoryTest(unittest.TestCase):
    def test_cage_fake_cache_summary_uses_packed_int2_payload_size(self):
        key_cache = CageKeyCache(
            key_quant_buckets=(
                torch.zeros(1, 2, 4, 1, dtype=torch.float16),
                torch.zeros(1, 2, 4, 3, dtype=torch.float16),
            ),
            key_full=torch.zeros(1, 2, 1, 4, dtype=torch.float16),
            key_bucket_indices=(torch.tensor([[0], [0]]), torch.tensor([[1, 2, 3], [1, 2, 3]])),
            key_group_sizes=(2, 4),
            key_clip_percentiles=(1.0, 1.0),
        )
        value_cache = CageValueCache(
            value_quant_buckets=(
                torch.zeros(1, 2, 4, 1, dtype=torch.float16),
                torch.zeros(1, 2, 4, 3, dtype=torch.float16),
            ),
            value_full=torch.zeros(1, 2, 1, 4, dtype=torch.float16),
            value_bucket_indices=(torch.tensor([[0], [0]]), torch.tensor([[1, 2, 3], [1, 2, 3]])),
            value_group_sizes=(2, 4),
            value_clip_percentiles=(1.0, 1.0),
        )
        cache = pack_cage_past_key_value(key_cache, value_cache)

        summary = summarize_cache_bytes(cache)

        self.assertEqual(summary["key_payload_bytes"], 8)
        self.assertEqual(summary["value_payload_bytes"], 8)
        self.assertLess(summary["key_payload_bytes"], key_cache.key_quant_buckets[0].nbytes + key_cache.key_quant_buckets[1].nbytes)
        self.assertEqual(summary["residual_full_precision_bytes"], 32)
        self.assertEqual(summary["bucket_index_bytes"], 128)
        json.dumps(summary)

    def test_parameterized_estimators_compare_kivi_and_cage_for_same_shape(self):
        kivi = estimate_kivi_cache_bytes(
            batch_size=1,
            num_key_value_heads=2,
            seq_len=5,
            head_dim=4,
            group_size=2,
            residual_length=1,
            bits=2,
        )
        cage = estimate_cage_cache_bytes(
            batch_size=1,
            num_key_value_heads=2,
            seq_len=5,
            head_dim=4,
            key_bucket_sizes=(1, 3),
            value_bucket_sizes=(1, 3),
            key_group_sizes=(2, 4),
            value_group_sizes=(2, 4),
            residual_length=1,
            bits=2,
        )

        self.assertEqual(kivi["residual_full_precision_bytes"], cage["residual_full_precision_bytes"])
        self.assertEqual(kivi["key_payload_bytes"], cage["key_payload_bytes"])
        self.assertGreater(cage["total_bytes"], cage["payload_only_bytes"])
        json.dumps(kivi)
        json.dumps(cage)

    def test_original_kivi_tuple_summary_uses_actual_cache_tensor_bytes(self):
        cache = (
            torch.zeros(1, 2, 4, 1, dtype=torch.int32),
            torch.zeros(1, 2, 1, 4, dtype=torch.float16),
            torch.zeros(1, 2, 4, 2, dtype=torch.float16),
            torch.zeros(1, 2, 4, 2, dtype=torch.float16),
            torch.zeros(1, 2, 4, 1, dtype=torch.int32),
            torch.zeros(1, 2, 1, 4, dtype=torch.float16),
            torch.zeros(1, 2, 4, 2, dtype=torch.float16),
            torch.zeros(1, 2, 4, 2, dtype=torch.float16),
            5,
        )

        summary = summarize_cache_bytes(cache)

        self.assertEqual(summary["key_payload_bytes"], cache[0].nbytes)
        self.assertEqual(summary["value_payload_bytes"], cache[4].nbytes)
        self.assertEqual(summary["key_scale_bytes"], cache[2].nbytes)
        self.assertEqual(summary["value_min_or_zp_bytes"], cache[7].nbytes)
        self.assertEqual(summary["bucket_index_bytes"], 0)
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
