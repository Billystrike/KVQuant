import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_analysis import (
    CANDIDATE,
    LENGTHS,
    METHODS,
    PREDECESSOR,
    UNIFORM_BASELINE,
    Llama2CageV3TransferQualityAnalysisError,
    build_analysis,
    validate_results_receipt,
    write_analysis_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_protocol_v1.json"
RECEIPT_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_results_receipt_v1.json"


class Llama2CageV3TransferQualityAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def _bytes(self):
        result = {}
        for row in self.protocol["method_length_matrix"]:
            length = row["prompt_length"]
            result[("fp16", length)] = length * 32 * 32 * 128 * 2 * 2
            result[("cage_v3", length)] = row["packed_memory"]["candidate_bytes"]
            result[("cage_v1", length)] = row["packed_memory"]["cage_v1_bytes"]
            result[("kivi", length)] = row["packed_memory"]["kivi_bytes"]
        return result

    def _records(self, overrides=None):
        values = {
            "fp16": {length: 1.99 for length in LENGTHS},
            CANDIDATE: {length: 2.0 for length in LENGTHS},
            PREDECESSOR: {length: 2.02 for length in LENGTHS},
            UNIFORM_BASELINE: {length: 2.02 for length in LENGTHS},
        }
        for method, by_length in (overrides or {}).items():
            values[method].update(by_length)
        packed = self._bytes()
        records = []
        for row in self.protocol["method_length_matrix"]:
            length = row["prompt_length"]
            for method_spec in row["methods"]:
                method = method_spec["method"]
                for anchor in range(50):
                    records.append({
                        "case_id": f"{method}-{length}-{anchor}",
                        "method": {"method": method},
                        "input": {"identity": {"prompt_length": length, "anchor_index": anchor}},
                        "memory": {"logical_packed_bytes": packed[(method, length)]},
                        "scoring": {"mean_nll": values[method][length] + anchor * 0.000001},
                    })
        return records

    def _analyze(self, records):
        return build_analysis(receipt_sha256="a" * 64, protocol=self.protocol, records=records)

    def test_checked_in_receipt_freezes_uninterpreted_results_and_exact_gate(self):
        validate_results_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        changed = copy.deepcopy(self.receipt)
        changed["analysis_freeze"]["overall_material_superiority_relative_ppl_percent_at_most"] = -0.1
        with self.assertRaises(Llama2CageV3TransferQualityAnalysisError):
            validate_results_receipt(changed, receipt_path=RECEIPT_PATH)

    def test_candidate_passes_only_when_both_primary_baselines_meet_all_rules(self):
        analysis = self._analyze(self._records())
        self.assertTrue(analysis["primary_transfer_gates"][PREDECESSOR]["pass"])
        self.assertTrue(analysis["primary_transfer_gates"][UNIFORM_BASELINE]["pass"])
        self.assertTrue(analysis["promotion_pass"])
        self.assertEqual(analysis["candidate_outcome"], "promote_cage_v3_as_llama2_main_method")

    def test_decimal_direction_without_material_superiority_is_not_a_pass(self):
        analysis = self._analyze(self._records({PREDECESSOR: {length: 2.005 for length in LENGTHS}}))
        gate = analysis["primary_transfer_gates"][PREDECESSOR]
        self.assertFalse(gate["overall_material_superiority_pass"])
        self.assertFalse(gate["pass"])
        self.assertFalse(analysis["promotion_pass"])

    def test_one_bad_length_fails_noninferiority(self):
        analysis = self._analyze(self._records({UNIFORM_BASELINE: {4032: 1.99}}))
        gate = analysis["primary_transfer_gates"][UNIFORM_BASELINE]
        self.assertFalse(gate["all_lengths_noninferiority_pass"])
        self.assertFalse(gate["pass"])

    def test_analysis_is_deterministic_complete_and_keeps_claim_boundaries(self):
        records = self._records()
        first = self._analyze(records)
        second = self._analyze(copy.deepcopy(records))
        self.assertEqual(first, second)
        self.assertEqual(len(first["method_length_summaries"]), 16)
        self.assertEqual(len(first["paired_comparisons"]), 12)
        self.assertEqual(first["bootstrap"]["resamples"], 10_000)
        self.assertFalse(first["next_authorization"]["paper_main_method_change"])
        self.assertFalse(first["next_authorization"]["candidate_tuning"])
        self.assertFalse(first["reporting"]["kitty_llama_included"])

    def test_grid_rejects_missing_case_or_changed_memory(self):
        records = self._records()
        with self.assertRaises(Llama2CageV3TransferQualityAnalysisError):
            self._analyze(records[:-1])
        changed = copy.deepcopy(records)
        changed[0]["memory"]["logical_packed_bytes"] += 1
        with self.assertRaisesRegex(Llama2CageV3TransferQualityAnalysisError, "packed bytes"):
            self._analyze(changed)

    def test_output_writer_handles_distinct_length_and_overall_fields(self):
        analysis = self._analyze(self._records())
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "analysis"
            outputs = write_analysis_outputs(analysis, destination)
            self.assertEqual(len(outputs), 4)
            self.assertTrue((destination / "llama2_cage_v3_paired_comparisons_v1.csv").is_file())
            self.assertIn("Frozen promotion decision", (destination / "llama2_cage_v3_transfer_decision_v1.md").read_text())


if __name__ == "__main__":
    unittest.main()
