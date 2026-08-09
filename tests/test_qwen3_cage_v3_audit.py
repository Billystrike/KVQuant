import copy
import json
import unittest
from pathlib import Path

from scripts.qwen3_validate_cage_v2_round1_decision import validate_decision
from utils.qwen3_cage_v3_audit import candidate_selection_audit, interior_anchor_starts


REPO_ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v2_round1_decision_v1.json"


class CageV3AuditTest(unittest.TestCase):
    def test_candidate_anchor_audits_are_deterministic_and_nonoverlapping(self):
        token_ids = list(range(500000))
        first = candidate_selection_audit(token_ids, 50)
        second = candidate_selection_audit(token_ids, 50)
        self.assertEqual(first, second)
        self.assertTrue(first["windows_nonoverlapping"])
        self.assertEqual(len(first["continuation_starts"]), 50)
        self.assertEqual(len(first["full_window_ids_sha256"]), 50)

    def test_anchor_audit_rejects_short_or_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            interior_anchor_starts(100, 15)
        with self.assertRaisesRegex(ValueError, ">= 2"):
            interior_anchor_starts(500000, 1)

    def test_negative_v2_decision_is_closed_and_machine_checkable(self):
        decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
        best = decision["frozen_outcome"]["best_track"]
        analysis = {
            "status": "pass",
            "claim_eligible": False,
            "advanced_family_count": 0,
            "advanced_families": [],
            "track_reports": [],
        }
        identities = [
            ("sr2-no-sink", "kitty-12.5pct"),
            ("sr2-no-sink", "kitty-pro-25pct"),
            ("sr1-sr2-no-sink", "kitty-12.5pct"),
            ("sr1-sr2-no-sink", "kitty-pro-25pct"),
            ("sr1-sr2-sink32", "kitty-12.5pct"),
            (best["family_id"], best["target"]),
        ]
        for family, target in identities:
            analysis["track_reports"].append({
                "family_id": family,
                "target": target,
                "gates": {"memory_all_three": True, "no_worse_than_kitty_at_least_two": False},
                "lengths": [
                    {"prompt_length": int(length), "versus_kitty": {"mean_delta": value}}
                    for length, value in best["length_mean_deltas_vs_kitty"].items()
                ],
                "overall_15_case_versus_kitty": {
                    "mean_delta": best["overall_15_case_mean_delta_vs_kitty"],
                    "favor_count": best["kitty_favor_count"],
                    "paired_count": best["paired_count"],
                },
                "overall_15_case_versus_cage_v1": {
                    "mean_delta": best["overall_15_case_mean_delta_vs_cage_v1"],
                    "favor_count": best["cage_v1_favor_count"],
                },
            })
        report = validate_decision(decision, analysis)
        self.assertEqual(report["status"], "pass")
        mutated = copy.deepcopy(analysis)
        mutated["advanced_family_count"] = 1
        with self.assertRaisesRegex(ValueError, "advanced family count"):
            validate_decision(decision, mutated)


if __name__ == "__main__":
    unittest.main()
