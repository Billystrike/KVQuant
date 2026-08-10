import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_holdout_analysis import (
    CageV3HoldoutAnalysisError,
    build_holdout_analysis,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_development_protocol_v1.json"
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_decision_v1.json"
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_postrun_receipt_v1.json"


class CageV3HoldoutAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def _records(self, candidate_by_length=None, control_by_length=None, kitty_by_length=None):
        candidate_by_length = candidate_by_length or {1024: 0.7, 2048: 0.7, 4032: 0.7}
        control_by_length = control_by_length or {1024: 1.0, 2048: 1.0, 4032: 1.0}
        kitty_by_length = kitty_by_length or {1024: 0.8, 2048: 0.8, 4032: 0.8}
        selected = self.decision["decision"]["selected_family_id"]
        family = next(row for row in self.protocol["candidate_families"] if row["family_id"] == selected)
        methods = []
        for point in family["points"]:
            methods.append((point["method_id"], selected, point["prompt_length"], point["packed_bytes"], candidate_by_length[point["prompt_length"]]))
        for point in self.protocol["cage_v2_best_controls"]:
            methods.append((point["method_id"], "cage-v2-best-control", point["prompt_length"], point["packed_bytes"], control_by_length[point["prompt_length"]]))
        for point in self.protocol["kitty_pro_targets"]:
            methods.append((point["method_id"], "kitty-pro-25pct", point["prompt_length"], point["target_bytes"], kitty_by_length[point["prompt_length"]]))
        records = []
        for method_id, family_id, length, packed_bytes, value in methods:
            for anchor in range(10, 20):
                records.append({
                    "method": {"id": method_id, "family_id": family_id},
                    "input": {"anchor_index": anchor, "prompt_length": length},
                    "memory": {"model_total_bytes": packed_bytes},
                    "aggregates": {"joint_post_o_proj_mse": {"mean": value}},
                })
        return records

    def _analyze(self, records):
        return build_holdout_analysis(
            protocol=self.protocol,
            decision=self.decision,
            receipt=self.receipt,
            receipt_sha256="a" * 64,
            records=records,
        )

    def test_preregistered_gate_passes_only_the_exact_three_conditions(self):
        analysis = self._analyze(self._records())
        self.assertTrue(analysis["holdout_gate_pass"])
        self.assertTrue(analysis["gates"]["memory_all_three"])
        self.assertTrue(analysis["gates"]["beats_cage_v2_all_three"])
        self.assertEqual(analysis["gates"]["kitty_pro_no_worse_length_count"], 3)
        self.assertEqual(len(analysis["lengths"]), 3)
        self.assertEqual(analysis["overall_30_case_versus_kitty_pro"]["paired_count"], 30)

    def test_cage_v2_threshold_is_strict_at_every_length(self):
        records = self._records(candidate_by_length={1024: 1.0, 2048: 0.7, 4032: 0.7})
        analysis = self._analyze(records)
        self.assertFalse(analysis["holdout_gate_pass"])
        self.assertFalse(analysis["gates"]["beats_cage_v2_all_three"])
        self.assertFalse(analysis["lengths"][0]["beats_cage_v2"])

    def test_kitty_threshold_is_nonpositive_at_two_of_three_lengths(self):
        records = self._records(candidate_by_length={1024: 0.8, 2048: 0.8, 4032: 0.9})
        analysis = self._analyze(records)
        self.assertTrue(analysis["holdout_gate_pass"])
        self.assertEqual(analysis["gates"]["kitty_pro_no_worse_length_count"], 2)
        self.assertFalse(analysis["lengths"][2]["no_worse_than_kitty_pro"])

    def test_bootstrap_cannot_enter_gate_and_reserved_anchors_are_rejected(self):
        records = self._records()
        analysis = self._analyze(records)
        self.assertFalse(analysis["bootstrap"]["used_for_gate"])
        self.assertFalse(analysis["bootstrap"]["descriptive_interval_computed"])
        changed = copy.deepcopy(records)
        changed[0]["input"]["anchor_index"] = 20
        with self.assertRaisesRegex(CageV3HoldoutAnalysisError, "holdout grid"):
            self._analyze(changed)

    def test_frozen_memory_is_reported_without_nominal_bit_shortcuts(self):
        analysis = self._analyze(self._records())
        expected = {row["prompt_length"]: row for row in self.protocol["kitty_pro_targets"]}
        for row in analysis["lengths"]:
            self.assertLessEqual(row["candidate_model_total_bytes"], expected[row["prompt_length"]]["target_bytes"])
            self.assertEqual(
                row["candidate_minus_kitty_pro_bytes"],
                row["candidate_model_total_bytes"] - row["kitty_pro_model_total_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
