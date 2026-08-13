import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_dtqi_analysis import (
    BASELINES,
    CANDIDATE,
    build_analysis,
    validate_results_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_results_receipt_v1.json"


class CageV4DTQIAnalysisTest(unittest.TestCase):
    def _records(self, candidate_delta=-0.02):
        records = []
        offsets = {
            CANDIDATE: candidate_delta,
            BASELINES[0]: 0.0,
            BASELINES[1]: 0.001,
        }
        bytes_by_method = {
            CANDIDATE: {1024: 45849600, 2048: 70050816, 4032: 110066688},
            BASELINES[0]: {1024: 47890080, 2048: 71782560, 4032: 110130336},
            BASELINES[1]: {1024: 45849600, 2048: 70050816, 4032: 110066688},
        }
        for method in (CANDIDATE, *BASELINES):
            for length in (1024, 2048, 4032):
                for document in range(20):
                    for anchor in (0, 1):
                        records.append({
                            "analysis_method_id": method,
                            "input": {
                                "document_id": f"document-{document}",
                                "anchor_index": anchor,
                                "prompt_length": length,
                            },
                            "memory": {"model_total_bytes": bytes_by_method[method][length]},
                            "scoring": {"mean_nll": 2.0 + offsets[method] + document * 0.0001 + anchor * 0.00001},
                        })
        return records

    def test_checked_in_receipt_freezes_success_policy_before_interpretation(self):
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        validate_results_receipt(receipt, receipt_path=RECEIPT)
        self.assertEqual(receipt["analysis_freeze"]["overall_material_superiority_relative_ppl_percent_at_most"], -1.0)
        self.assertEqual(receipt["analysis_freeze"]["per_length_noninferiority_relative_ppl_percent_at_most"], 0.5)
        self.assertFalse(receipt["authorization"]["holdout_metrics"])

    def test_receipt_rejects_relaxed_threshold_or_holdout_access(self):
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        changed = copy.deepcopy(receipt)
        changed["analysis_freeze"]["overall_material_superiority_relative_ppl_percent_at_most"] = -0.1
        with self.assertRaisesRegex(RuntimeError, "analysis freeze changed"):
            validate_results_receipt(changed, receipt_path=RECEIPT)
        changed = copy.deepcopy(receipt)
        changed["authorization"]["holdout_metrics"] = True
        with self.assertRaisesRegex(RuntimeError, "authorization changed"):
            validate_results_receipt(changed, receipt_path=RECEIPT)

    def test_synthetic_material_candidate_advances_only_when_both_baselines_pass(self):
        analysis = build_analysis(receipt_sha256="a" * 64, records=self._records(-0.02))
        self.assertEqual(len(analysis["method_length_summaries"]), 9)
        self.assertEqual(len(analysis["paired_comparisons"]), 8)
        self.assertTrue(analysis["success_decision"]["all_two_baselines_pass"])
        self.assertEqual(analysis["success_decision"]["candidate_outcome"], "advance_to_holdout_preflight")
        self.assertFalse(analysis["boundaries"]["holdout_accessed"])

    def test_small_decimal_improvement_closes_candidate(self):
        analysis = build_analysis(receipt_sha256="b" * 64, records=self._records(-0.001))
        self.assertFalse(analysis["success_decision"]["all_two_baselines_pass"])
        self.assertEqual(analysis["success_decision"]["candidate_outcome"], "close_cage_v4_as_negative")

    def test_analysis_runner_cannot_access_holdout_or_test(self):
        source = (REPO_ROOT / "scripts" / "qwen3_analyze_cage_v4_dtqi_screen.py").read_text(encoding="utf-8")
        self.assertNotIn('"--stage"', source)
        self.assertNotIn('"--partition"', source)
        self.assertNotIn("holdout_full", source.lower())
        self.assertNotIn("holdout_execution", source.lower())
        self.assertNotIn("local_perturbation", source)


if __name__ == "__main__":
    unittest.main()
