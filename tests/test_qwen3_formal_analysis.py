import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.qwen3_formal import formal_method_length_points, load_formal_protocol
from utils.qwen3_formal_analysis import (
    Qwen3FormalAnalysisError,
    build_formal_analysis,
    packed_bytes_for_method,
    paired_delta_statistics,
    validate_results_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


class Qwen3FormalAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(
            (ROOT / "configs" / "qwen3_8b_formal_results_receipt_v1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.protocol, _ = load_formal_protocol(
            ROOT / "configs" / "qwen3_8b_formal_quality_protocol_v1.json"
        )
        cls.memory_protocol = json.loads(
            (ROOT / "configs" / "qwen3_8b_matched_memory_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_checked_in_results_receipt_is_frozen_before_interpretation(self):
        validate_results_receipt(self.receipt)
        self.assertEqual(self.receipt["partitions"]["cage_qwen3"]["case_count"], 1000)
        self.assertEqual(self.receipt["partitions"]["kitty_qwen3"]["case_count"], 300)
        self.assertEqual(
            self.receipt["analysis_freeze"]["bootstrap_interval"],
            "two_sided_95_percentile_type7_linear",
        )

    def test_receipt_rejects_posthoc_bootstrap_mutation(self):
        mutated = copy.deepcopy(self.receipt)
        mutated["analysis_freeze"]["bootstrap_resamples"] = 9999
        with self.assertRaisesRegex(Qwen3FormalAnalysisError, "analysis freeze"):
            validate_results_receipt(mutated)

    def test_paired_statistics_use_exact_direction_counts_and_are_deterministic(self):
        baseline = {anchor: 2.0 for anchor in range(50)}
        candidate = {
            anchor: 1.0 if anchor < 20 else 2.0 if anchor < 30 else 3.0
            for anchor in range(50)
        }
        left = paired_delta_statistics(
            candidate,
            baseline,
            bootstrap_resamples=200,
            bootstrap_seed=17,
        )
        right = paired_delta_statistics(
            candidate,
            baseline,
            bootstrap_resamples=200,
            bootstrap_seed=17,
        )
        self.assertEqual(left, right)
        self.assertEqual(left["favor_count"], 20)
        self.assertEqual(left["tie_count"], 10)
        self.assertEqual(left["oppose_count"], 20)
        self.assertEqual(left["mean_delta_nll"], 0.0)
        self.assertEqual(left["exp_of_mean_delta"], 1.0)

    def test_packed_bytes_reproduce_each_method_family(self):
        points = {
            (point["method_id"], point["prompt_length"]): point
            for partition in ("cage_qwen3", "kitty_qwen3")
            for point in formal_method_length_points(self.protocol, partition=partition)
        }
        expected = {
            ("fp16", 1024): 150994944,
            ("cage-r224", 1024): 48253824,
            ("kivi-g64-r320", 1024): 47480832,
            ("kitty-12.5pct", 1024): 46857888,
            ("kitty-pro-25pct", 1024): 47890080,
        }
        for key, byte_total in expected.items():
            point = points[key]
            method = {
                "id": point["method_id"],
                "name": point["method"],
                "config": point["config"],
            }
            with self.subTest(point=key):
                self.assertEqual(packed_bytes_for_method(method, key[1]), byte_total)

    def test_builds_complete_synthetic_analysis_without_changing_frozen_grid(self):
        records = []
        point_order = 0
        for partition in ("cage_qwen3", "kitty_qwen3"):
            for point in formal_method_length_points(self.protocol, partition=partition):
                method = {
                    "id": point["method_id"],
                    "name": point["method"],
                    "config": point["config"],
                }
                base = 2.0 + point_order * 0.01
                for anchor in range(50):
                    mean = base + anchor * 0.0001
                    records.append(
                        {
                            "method": method,
                            "input": {
                                "prompt_length": point["prompt_length"],
                                "anchor_index": anchor,
                            },
                            "scoring": {
                                "token_nlls": [mean] * 64,
                                "mean_nll": mean,
                            },
                        }
                    )
                point_order += 1

        original = paired_delta_statistics

        def fast_statistics(candidate, baseline, **kwargs):
            return original(
                candidate,
                baseline,
                bootstrap_resamples=20,
                bootstrap_seed=kwargs["bootstrap_seed"],
            )

        with patch(
            "utils.qwen3_formal_analysis.paired_delta_statistics",
            side_effect=fast_statistics,
        ):
            analysis = build_formal_analysis(
                protocol=self.protocol,
                memory_protocol=self.memory_protocol,
                receipt=self.receipt,
                records=records,
                receipt_sha256="a" * 64,
            )

        self.assertEqual(analysis["status"], "pass")
        self.assertEqual(analysis["case_count"], 1300)
        self.assertEqual(analysis["target_count"], 83200)
        self.assertEqual(analysis["method_length_point_count"], 26)
        self.assertEqual(analysis["base_method_length_point_count"], 18)
        self.assertEqual(len(analysis["primary_matched_comparisons"]), 10)
        self.assertEqual(len(analysis["mechanism_comparisons"]), 12)
        self.assertEqual(len(analysis["base_pareto_by_prompt_length"]), 3)
        self.assertTrue(
            all(
                record["candidate_packed_bytes"] == record["baseline_packed_bytes"]
                for record in analysis["mechanism_comparisons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
