import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_holdout_decision import (
    CageV3HoldoutDecisionError,
    EXPECTED_HOLDOUT_DECISION_SHA256,
    validate_holdout_decision,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_decision_v1.json"


class CageV3HoldoutDecisionTest(unittest.TestCase):
    def setUp(self):
        self.decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

    def test_negative_holdout_decision_is_exactly_frozen(self):
        self.assertEqual(file_sha256(DECISION_PATH), EXPECTED_HOLDOUT_DECISION_SHA256)
        validate_holdout_decision(self.decision, decision_path=DECISION_PATH)
        self.assertFalse(self.decision["decision"]["holdout_gate_pass"])
        self.assertEqual(self.decision["decision"]["gates"]["kitty_pro_no_worse_length_count"], 1)
        self.assertEqual(
            [row["no_worse_than_kitty_pro"] for row in self.decision["all_lengths"]],
            [True, False, False],
        )

    def test_negative_decision_cannot_authorize_followup_or_reserved_data(self):
        changed = copy.deepcopy(self.decision)
        changed["closure"]["kitty_12_5pct_followup_authorized"] = True
        with self.assertRaisesRegex(CageV3HoldoutDecisionError, "closure"):
            validate_holdout_decision(changed, decision_path=DECISION_PATH)
        changed = copy.deepcopy(self.decision)
        changed["consumption_state"]["reserved_unseen_metrics_consumed"] = True
        with self.assertRaisesRegex(CageV3HoldoutDecisionError, "consumption"):
            validate_holdout_decision(changed, decision_path=DECISION_PATH)

    def test_frozen_analysis_sources_match_the_interpreting_commit(self):
        expected = self.decision["frozen_inputs"]["analysis_source_sha256"]
        paths = {
            "script": REPO_ROOT / "scripts" / "qwen3_analyze_cage_v3_holdout.py",
            "utils": REPO_ROOT / "utils" / "qwen3_cage_v3_holdout_analysis.py",
            "tests": REPO_ROOT / "tests" / "test_qwen3_cage_v3_holdout_analysis.py",
        }
        for name, path in paths.items():
            self.assertEqual(file_sha256(path), expected[name])


if __name__ == "__main__":
    unittest.main()
