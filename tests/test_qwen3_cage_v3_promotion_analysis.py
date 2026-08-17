import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_analysis import (
    CANDIDATE,
    EXTERNAL_BASELINE,
    PREDECESSOR,
    UNIFORM_BASELINE,
    CageV3PromotionAnalysisError,
    build_promotion_analysis,
    validate_results_receipt,
)
from utils.qwen3_cage_v3_promotion_protocol import METHOD_IDS, PROMPT_LENGTHS


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_protocol_v1.json"
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_results_receipt_v1.json"


class CageV3PromotionAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def _records(self, overrides=None):
        values = {
            "fp16": {length: 1.99 for length in PROMPT_LENGTHS},
            UNIFORM_BASELINE: {length: 2.02 for length in PROMPT_LENGTHS},
            PREDECESSOR: {length: 2.001 for length in PROMPT_LENGTHS},
            CANDIDATE: {length: 2.0 for length in PROMPT_LENGTHS},
            EXTERNAL_BASELINE: {length: 2.0 for length in PROMPT_LENGTHS},
        }
        for method, by_length in (overrides or {}).items():
            values[method].update(by_length)
        frozen_bytes = {
            (method["method_id"], point["prompt_length"]): point["packed_bytes"]
            for method in self.protocol["method_grid"]
            for point in method["points"]
        }
        records = []
        for method in METHOD_IDS:
            for length in PROMPT_LENGTHS:
                for document_index in range(20):
                    document = f"pg19-document-{document_index:02d}"
                    document_effect = document_index * 0.00001
                    for anchor in (0, 1):
                        records.append(
                            {
                                "case_id": f"{method}-{length}-{document_index:02d}-{anchor}",
                                "method": {"metric_method_id": method},
                                "input": {
                                    "prompt_length": length,
                                    "document_id": document,
                                    "anchor_index": anchor,
                                },
                                "memory": {
                                    "model_total_bytes": frozen_bytes[(method, length)]
                                },
                                "scoring": {
                                    "mean_nll": values[method][length] + document_effect
                                },
                            }
                        )
        return records

    def _analyze(self, records):
        return build_promotion_analysis(
            receipt_sha256="a" * 64,
            protocol=self.protocol,
            records=records,
        )

    def test_checked_in_receipt_freezes_uninterpreted_results_and_boundaries(self):
        validate_results_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        changed = copy.deepcopy(self.receipt)
        changed["authorization"]["pg19_test_access"] = True
        with self.assertRaisesRegex(CageV3PromotionAnalysisError, "differs from frozen"):
            validate_results_receipt(changed, receipt_path=RECEIPT_PATH)

    def test_promotion_can_pass_predecessor_pareto_track_and_both_primary_baselines(self):
        analysis = self._analyze(self._records())
        predecessor = analysis["promotion_gates"]["versus_predecessor"]
        self.assertFalse(predecessor["quality_superiority_track_pass"])
        self.assertTrue(predecessor["memory_quality_pareto_track_pass"])
        self.assertTrue(predecessor["pass"])
        self.assertTrue(analysis["promotion_gates"]["versus_uniform_baseline"]["pass"])
        self.assertTrue(analysis["promotion_gates"]["versus_external_baseline"]["pass"])
        self.assertTrue(analysis["promotion_pass"])
        self.assertTrue(analysis["next_authorization"]["staged_llama2_cpu_gpu_acceptance"])
        self.assertFalse(analysis["next_authorization"]["llama2_full_experiments"])
        self.assertFalse(analysis["next_authorization"]["kitty_llama_port"])

    def test_uniform_gate_requires_material_superiority_not_decimal_direction(self):
        records = self._records({UNIFORM_BASELINE: {length: 2.005 for length in PROMPT_LENGTHS}})
        analysis = self._analyze(records)
        uniform = analysis["promotion_gates"]["versus_uniform_baseline"]
        self.assertTrue(uniform["all_lengths_noninferiority_pass"])
        self.assertFalse(uniform["pass"])
        self.assertFalse(analysis["promotion_pass"])

    def test_external_gate_rejects_one_length_beyond_half_percent(self):
        records = self._records({EXTERNAL_BASELINE: {4032: 1.99}})
        analysis = self._analyze(records)
        external = analysis["promotion_gates"]["versus_external_baseline"]
        self.assertFalse(external["all_lengths_noninferiority_pass"])
        self.assertFalse(external["pass"])
        self.assertFalse(analysis["promotion_pass"])

    def test_predecessor_gate_rejects_quality_loss_despite_frozen_memory_saving(self):
        records = self._records({PREDECESSOR: {length: 1.99 for length in PROMPT_LENGTHS}})
        analysis = self._analyze(records)
        predecessor = analysis["promotion_gates"]["versus_predecessor"]
        self.assertFalse(predecessor["quality_superiority_track_pass"])
        self.assertFalse(predecessor["memory_quality_pareto_track_pass"])
        self.assertFalse(predecessor["pass"])
        self.assertFalse(analysis["promotion_pass"])

    def test_analysis_is_deterministic_and_reports_every_method_length(self):
        records = self._records()
        first = self._analyze(records)
        second = self._analyze(copy.deepcopy(records))
        self.assertEqual(first, second)
        self.assertEqual(len(first["method_length_summaries"]), 15)
        self.assertEqual(len(first["candidate_comparisons"]), 16)
        self.assertEqual(first["bootstrap"]["resamples"], 10_000)
        self.assertEqual(first["bootstrap"]["seed"], 20_260_816)
        self.assertFalse(first["next_authorization"]["pg19_test_access"])
        self.assertFalse(first["next_authorization"]["paper_claims"])
        self.assertTrue(first["reporting"]["all_methods_lengths_and_unfavorable_results_retained"])

    def test_grid_rejects_missing_anchor_or_changed_packed_bytes(self):
        records = self._records()
        with self.assertRaises(CageV3PromotionAnalysisError):
            self._analyze(records[:-1])
        changed = copy.deepcopy(records)
        changed[0]["memory"]["model_total_bytes"] += 1
        with self.assertRaisesRegex(CageV3PromotionAnalysisError, "packed bytes"):
            self._analyze(changed)


if __name__ == "__main__":
    unittest.main()
