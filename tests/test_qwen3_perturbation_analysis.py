import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.qwen3_formal import formal_method_length_points, load_formal_protocol
from utils.qwen3_formal_analysis import packed_bytes_for_method
from utils.qwen3_perturbation_analysis import (
    Qwen3PerturbationAnalysisError,
    build_perturbation_analysis,
    paired_delta_statistics,
    validate_results_receipt,
)
from utils.qwen3_perturbation_protocol import (
    LAYER_METRICS,
    aggregate_layer_metrics,
    load_perturbation_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class Qwen3PerturbationAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(
            (
                ROOT
                / "configs"
                / "qwen3_8b_memory_perturbation_results_receipt_v1.json"
            ).read_text(encoding="utf-8")
        )
        cls.quality, _ = load_formal_protocol(
            ROOT / "configs" / "qwen3_8b_formal_quality_protocol_v1.json"
        )
        cls.perturbation, _ = load_perturbation_protocol(
            ROOT / "configs" / "qwen3_8b_memory_perturbation_protocol_v1.json"
        )

    def test_checked_in_receipt_precedes_perturbation_interpretation(self):
        validate_results_receipt(self.receipt)
        self.assertFalse(
            self.receipt["joint_postrun_audit"]["interpretation_performed"]
        )
        self.assertEqual(self.receipt["joint_postrun_audit"]["case_count"], 1300)
        self.assertEqual(
            self.receipt["joint_postrun_audit"]["layer_record_count"], 46800
        )
        self.assertEqual(
            self.receipt["analysis_freeze"]["primary_metric"],
            "joint_post_o_proj_mse",
        )

    def test_receipt_rejects_posthoc_primary_metric_mutation(self):
        mutated = copy.deepcopy(self.receipt)
        mutated["analysis_freeze"]["primary_metric"] = "post_o_proj_mse"
        with self.assertRaisesRegex(Qwen3PerturbationAnalysisError, "analysis freeze"):
            validate_results_receipt(mutated)

    def test_paired_statistics_are_directional_and_deterministic(self):
        baseline = {anchor: 2.0 for anchor in range(50)}
        candidate = {
            anchor: 1.0 if anchor < 20 else 2.0 if anchor < 30 else 3.0
            for anchor in range(50)
        }
        left = paired_delta_statistics(
            candidate, baseline, bootstrap_resamples=200, bootstrap_seed=17
        )
        right = paired_delta_statistics(
            candidate, baseline, bootstrap_resamples=200, bootstrap_seed=17
        )
        self.assertEqual(left, right)
        self.assertEqual(left["favor_count"], 20)
        self.assertEqual(left["tie_count"], 10)
        self.assertEqual(left["oppose_count"], 20)
        self.assertEqual(left["mean_delta"], 0.0)

    def test_builds_complete_synthetic_analysis_on_the_frozen_grid(self):
        records = []
        point_index = 0
        for partition in ("cage_qwen3", "kitty_qwen3"):
            for point in formal_method_length_points(self.quality, partition=partition):
                method = {
                    "id": point["method_id"],
                    "name": point["method"],
                    "config": point["config"],
                }
                packed_bytes = packed_bytes_for_method(method, point["prompt_length"])
                for anchor in range(50):
                    layers = []
                    for layer_idx in range(36):
                        value = point_index * 1e-5 + anchor * 1e-7 + layer_idx * 1e-9
                        metrics = {name: value for name in LAYER_METRICS}
                        metrics["topk_attention_overlap"] = 1.0 - value
                        layers.append({"layer_idx": layer_idx, "metrics": metrics})
                    records.append(
                        {
                            "method": method,
                            "input": {
                                "prompt_length": point["prompt_length"],
                                "anchor_index": anchor,
                            },
                            "memory": {"model_total_bytes": packed_bytes},
                            "layer_metrics": layers,
                            "aggregates": aggregate_layer_metrics(layers),
                        }
                    )
                point_index += 1

        original = paired_delta_statistics

        def fast_statistics(candidate, baseline, **kwargs):
            return original(
                candidate,
                baseline,
                bootstrap_resamples=20,
                bootstrap_seed=kwargs["bootstrap_seed"],
            )

        with patch(
            "utils.qwen3_perturbation_analysis.paired_delta_statistics",
            side_effect=fast_statistics,
        ):
            analysis = build_perturbation_analysis(
                perturbation_protocol=self.perturbation,
                quality_protocol=self.quality,
                receipt=self.receipt,
                records=records,
                receipt_sha256="a" * 64,
            )

        self.assertEqual(analysis["status"], "pass")
        self.assertEqual(analysis["case_count"], 1300)
        self.assertEqual(analysis["layer_record_count"], 46800)
        self.assertEqual(analysis["method_length_point_count"], 26)
        self.assertEqual(analysis["base_method_length_point_count"], 18)
        self.assertEqual(len(analysis["primary_matched_comparisons"]), 10)
        self.assertEqual(len(analysis["mechanism_comparisons"]), 12)
        self.assertEqual(len(analysis["layer_summaries"]), 26 * 36)
        self.assertEqual(len(analysis["base_pareto_by_prompt_length"]), 3)
        self.assertTrue(
            all(
                record["candidate_packed_bytes"]
                == record["baseline_packed_bytes"]
                for record in analysis["mechanism_comparisons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
