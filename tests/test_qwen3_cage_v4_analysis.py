import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_analysis import (
    ALL_METHODS,
    CageV4AnalysisError,
    bootstrap_draws,
    build_metric_screen_analysis,
    directional_concordance,
    practical_effect,
    spearman_correlation,
    validate_results_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = (
    REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_results_receipt_v1.json"
)


class CageV4AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def _synthetic_records(self):
        records = []
        degradation = {method: index * 0.01 for index, method in enumerate(ALL_METHODS)}
        proxy = {method: index * 0.001 for index, method in enumerate(ALL_METHODS)}
        for method_index, method in enumerate(ALL_METHODS):
            for length in (1024, 2048, 4032):
                for document_index in range(20):
                    document = f"document-{document_index:02d}"
                    document_effect = document_index * 0.0001
                    for anchor in (0, 1):
                        local = None
                        if method != "fp16":
                            local = {
                                "aggregates": {
                                    "joint_post_o_proj_mse": {
                                        "mean": proxy[method] + length / 1_000_000_000
                                    }
                                }
                            }
                        records.append(
                            {
                                "method": {
                                    "metric_method_id": method,
                                },
                                "input": {
                                    "prompt_length": length,
                                    "document_id": document,
                                    "anchor_index": anchor,
                                },
                                "memory": {
                                    "model_total_bytes": 10_000_000 + method_index * 1000 + length
                                },
                                "scoring": {
                                    "mean_nll": 2.0 + degradation[method] + document_effect
                                },
                                "local_perturbation": local,
                            }
                        )
        return records

    def test_receipt_freezes_analysis_before_interpretation_and_blocks_holdout(self):
        validate_results_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        changed = copy.deepcopy(self.receipt)
        changed["authorization"]["holdout_metrics"] = True
        with self.assertRaisesRegex(CageV4AnalysisError, "boundary"):
            validate_results_receipt(changed, receipt_path=RECEIPT_PATH)

    def test_spearman_average_ranks_and_directional_ties_are_deterministic(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]), 1.0)
        report = directional_concordance(
            {"a": 1.0, "b": 1.0, "c": 2.0},
            {"a": 3.0, "b": 3.0, "c": 1.0},
            ("a", "b", "c"),
        )
        self.assertEqual(report["pair_count"], 3)
        self.assertEqual(report["concordant_count"], 1)

    def test_practical_effect_uses_frozen_half_and_one_percent_thresholds(self):
        self.assertEqual(practical_effect(0.49)["magnitude_label"], "practically_equivalent")
        self.assertIn("small_effect", practical_effect(-0.5)["magnitude_label"])
        self.assertEqual(practical_effect(1.0)["magnitude_label"], "material_effect")
        self.assertEqual(practical_effect(-1.0)["direction"], "candidate_better")

    def test_bootstrap_draws_are_seeded_and_reused(self):
        first = bootstrap_draws(20, resamples=10, seed=20_260_810)
        second = bootstrap_draws(20, resamples=10, seed=20_260_810)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(len(draw) == 20 for draw in first))

    def test_synthetic_analysis_reports_every_point_and_passes_aligned_proxy(self):
        analysis = build_metric_screen_analysis(
            receipt_sha256="a" * 64,
            records=self._synthetic_records(),
        )
        self.assertEqual(len(analysis["method_length_summaries"]), 18)
        self.assertEqual(len(analysis["paired_nll_comparisons"]), 60)
        self.assertTrue(analysis["local_proxy_validity"]["gate_pass"])
        self.assertEqual(
            analysis["local_proxy_validity"]["frontier_pair_concordant_count"], 9
        )
        self.assertFalse(analysis["boundaries"]["candidate_selection_performed"])
        self.assertFalse(analysis["boundaries"]["holdout_accessed"])


if __name__ == "__main__":
    unittest.main()
