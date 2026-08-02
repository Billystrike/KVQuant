import json
import unittest

import torch

from models.cage_cache import CageKeyCache, CageValueCache, pack_cage_past_key_value
from utils.cage_memory import (
    build_memory_namespace,
    estimate_cage_cache_bytes,
    estimate_kivi_cache_bytes,
    sum_cache_summaries,
    summarize_cache_bytes,
    summarize_fp16_cache_bytes,
    summarize_cache_structure,
    summarize_runtime_cache_bytes,
)


class CageMemoryTest(unittest.TestCase):
    def _cage_cache(self):
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
        return pack_cage_past_key_value(key_cache, value_cache)

    def test_cage_runtime_summary_counts_fake_tensors(self):
        cache = self._cage_cache()

        paper = summarize_cache_bytes(cache)
        runtime = summarize_runtime_cache_bytes(cache)

        self.assertGreater(runtime["total_bytes"], paper["payload_only_bytes"])
        self.assertEqual(runtime["key_payload_bytes"], 64)
        self.assertEqual(runtime["value_payload_bytes"], 64)
        self.assertEqual(runtime["bucket_index_bytes"], 128)
        self.assertEqual(runtime["cache_type"], "cage_fake_runtime")

    def test_fp16_summary_counts_key_and_value_as_full_precision(self):
        key = torch.zeros(1, 2, 5, 4, dtype=torch.float16)
        value = torch.zeros(1, 2, 5, 4, dtype=torch.float16)

        summary = summarize_fp16_cache_bytes((key, value))

        self.assertEqual(summary["residual_full_precision_bytes"], key.nbytes + value.nbytes)
        self.assertEqual(summary["payload_only_bytes"], 0)
        self.assertEqual(summary["metadata_bytes"], 0)
        self.assertEqual(summary["cache_type"], "fp16")

    def test_cache_structure_reports_fp16_kivi_and_cage_key_value_tokens(self):
        fp16 = (
            torch.zeros(1, 2, 5, 4, dtype=torch.float16),
            torch.zeros(1, 2, 5, 4, dtype=torch.float16),
        )
        self.assertEqual(
            summarize_cache_structure("fp16", fp16, prompt_length=5),
            {
                "key": {"total_tokens": 5, "quantized_history_tokens": 0,
                        "fp16_residual_tokens": 5},
                "value": {"total_tokens": 5, "quantized_history_tokens": 0,
                          "fp16_residual_tokens": 5},
            },
        )

        kivi = (
            torch.zeros(1), torch.zeros(1, 2, 1, 4), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1, 2, 2, 4), torch.zeros(1), torch.zeros(1), 5,
        )
        expected_quantized = {
            "key": {"total_tokens": 5, "quantized_history_tokens": 4,
                    "fp16_residual_tokens": 1},
            "value": {"total_tokens": 5, "quantized_history_tokens": 3,
                      "fp16_residual_tokens": 2},
        }
        self.assertEqual(
            summarize_cache_structure("kivi", kivi, prompt_length=5, residual_length=2),
            expected_quantized,
        )

        key_cache = CageKeyCache(
            key_quant_buckets=(torch.zeros(1, 2, 4, 4),),
            key_full=torch.zeros(1, 2, 1, 4),
            key_bucket_indices=(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),),
            key_group_sizes=(2,), key_clip_percentiles=(1.0,),
        )
        value_cache = CageValueCache(
            value_quant_buckets=(torch.zeros(1, 2, 3, 4),),
            value_full=torch.zeros(1, 2, 2, 4),
            value_bucket_indices=(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),),
            value_group_sizes=(2,), value_clip_percentiles=(1.0,),
        )
        cage = pack_cage_past_key_value(key_cache, value_cache, kv_seq_len=5)
        self.assertEqual(
            summarize_cache_structure("cage", cage, prompt_length=5, residual_length=2),
            expected_quantized,
        )

    def test_cache_structure_rejects_semantically_wrong_residuals(self):
        invalid = (
            torch.zeros(1), torch.zeros(1, 2, 2, 4), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1, 2, 2, 4), torch.zeros(1), torch.zeros(1), 5,
        )
        with self.assertRaisesRegex(ValueError, "Key residual"):
            summarize_cache_structure("kivi", invalid, prompt_length=5, residual_length=2)

    def test_sum_cache_summaries_adds_every_byte_field(self):
        total = sum_cache_summaries([
            {"total_bytes": 10, "payload_only_bytes": 4, "metadata_bytes": 3},
            {"total_bytes": 20, "payload_only_bytes": 8, "bucket_index_bytes": 7},
        ])

        self.assertEqual(total["total_bytes"], 30)
        self.assertEqual(total["payload_only_bytes"], 12)
        self.assertEqual(total["metadata_bytes"], 3)
        self.assertEqual(total["bucket_index_bytes"], 7)
        self.assertEqual(total["cache_type"], "model_total")

    def test_build_memory_namespace_uses_exact_worker_keys(self):
        paper = {"cache_type": "model_total", "total_bytes": 10}
        runtime = {"cache_type": "model_total", "total_bytes": 20}

        memory = build_memory_namespace(
            paper,
            runtime,
            max_allocated_bytes=30,
            max_reserved_bytes=40,
        )

        self.assertEqual(
            memory,
            {
                "paper_estimate": paper,
                "runtime_tensors": runtime,
                "cuda_peak_diagnostic": {
                    "max_allocated_bytes": 30,
                    "max_reserved_bytes": 40,
                },
            },
        )

    def test_paper_summary_exposes_metadata_separately_from_bucket_indices(self):
        summary = summarize_cache_bytes(self._cage_cache())

        self.assertIn("metadata_bytes", summary)
        self.assertIn("bucket_index_bytes", summary)
        self.assertEqual(
            summary["metadata_bytes"],
            summary["key_scale_bytes"]
            + summary["value_scale_bytes"]
            + summary["key_min_or_zp_bytes"]
            + summary["value_min_or_zp_bytes"],
        )
        self.assertEqual(
            summary["total_bytes"],
            summary["payload_only_bytes"]
            + summary["metadata_bytes"]
            + summary["bucket_index_bytes"]
            + summary["residual_full_precision_bytes"],
        )

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
