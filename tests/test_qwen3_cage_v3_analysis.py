import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_analysis import (
    CageV3AnalysisError,
    build_screen_analysis,
    paired_summary,
    validate_screen_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_postrun_receipt_v1.json"
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_development_protocol_v1.json"


class CageV3AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def _synthetic_records(self, candidate_values):
        records = []
        methods = []
        for point in self.protocol["kitty_pro_targets"]:
            methods.append((point["method_id"], "kitty-pro-25pct", point["prompt_length"], point["target_bytes"], 0.8))
        for point in self.protocol["cage_v2_best_controls"]:
            methods.append((point["method_id"], "cage-v2-best-control", point["prompt_length"], point["packed_bytes"], 1.0))
        for family in self.protocol["candidate_families"]:
            for point in family["points"]:
                methods.append((
                    point["method_id"],
                    family["family_id"],
                    point["prompt_length"],
                    point["packed_bytes"],
                    candidate_values[family["family_id"]],
                ))
        for method_id, family_id, length, packed_bytes, value in methods:
            for anchor in range(5, 10):
                records.append({
                    "method": {"id": method_id, "family_id": family_id},
                    "input": {"anchor_index": anchor, "prompt_length": length},
                    "memory": {"model_total_bytes": packed_bytes},
                    "aggregates": {"joint_post_o_proj_mse": {"mean": value}},
                })
        return records

    def test_receipt_is_exactly_the_preinterpretation_freeze(self):
        validate_screen_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        changed = copy.deepcopy(self.receipt)
        changed["next_authorization"]["holdout_metrics"] = True
        with self.assertRaisesRegex(CageV3AnalysisError, "authorization"):
            validate_screen_receipt(changed, receipt_path=RECEIPT_PATH)

    def test_paired_summary_is_directional_and_exact(self):
        candidate = {anchor: 0.7 for anchor in range(5, 10)}
        baseline = {anchor: 0.8 for anchor in range(5, 10)}
        summary = paired_summary(candidate, baseline)
        self.assertLess(summary["mean_delta"], 0)
        self.assertEqual(summary["favor_count"], 5)
        self.assertEqual(summary["paired_count"], 5)

    def test_gate_selects_at_most_one_with_frozen_tie_break(self):
        families = [row["family_id"] for row in self.protocol["candidate_families"]]
        records = self._synthetic_records({families[0]: 0.7, families[1]: 0.75, families[2]: 0.9})
        analysis = build_screen_analysis(
            protocol=self.protocol,
            receipt=self.receipt,
            receipt_sha256="a" * 64,
            records=records,
        )
        self.assertEqual(analysis["advanced_family_count"], 1)
        self.assertEqual(analysis["advanced_families"][0]["family_id"], families[0])
        self.assertTrue(analysis["holdout_authorized_after_decision_freeze"])
        self.assertEqual(len(analysis["family_reports"]), 3)

    def test_none_pass_closes_screen_without_holdout(self):
        families = [row["family_id"] for row in self.protocol["candidate_families"]]
        records = self._synthetic_records({family: 1.1 for family in families})
        analysis = build_screen_analysis(
            protocol=self.protocol,
            receipt=self.receipt,
            receipt_sha256="b" * 64,
            records=records,
        )
        self.assertEqual(analysis["advanced_family_count"], 0)
        self.assertFalse(analysis["holdout_authorized_after_decision_freeze"])
        self.assertIn("closed_negative", analysis["decision_status"])

    def test_gate_uses_strict_control_and_nonpositive_kitty_thresholds(self):
        families = [row["family_id"] for row in self.protocol["candidate_families"]]
        records = self._synthetic_records({families[0]: 0.8, families[1]: 1.0, families[2]: 1.1})
        analysis = build_screen_analysis(
            protocol=self.protocol,
            receipt=self.receipt,
            receipt_sha256="c" * 64,
            records=records,
        )
        reports = {row["family_id"]: row for row in analysis["family_reports"]}
        self.assertTrue(reports[families[0]]["family_pass"])
        self.assertEqual(
            reports[families[0]]["gates"]["kitty_pro_no_worse_length_count"],
            3,
        )
        self.assertFalse(reports[families[1]]["gates"]["beats_cage_v2_all_three"])

    def test_equal_quality_and_bytes_fall_back_to_family_id(self):
        families = [row["family_id"] for row in self.protocol["candidate_families"]]
        records = self._synthetic_records({families[0]: 0.7, families[1]: 0.7, families[2]: 0.9})
        analysis = build_screen_analysis(
            protocol=self.protocol,
            receipt=self.receipt,
            receipt_sha256="d" * 64,
            records=records,
        )
        expected = min(families[0], families[1])
        self.assertEqual(analysis["advanced_families"][0]["family_id"], expected)


if __name__ == "__main__":
    unittest.main()
