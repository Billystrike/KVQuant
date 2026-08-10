import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.qwen3_compare_cage_v3_holdout_acceptance import compare_repeats
from utils.qwen3_cage_v3_holdout import (
    CageV3HoldoutError,
    EXPECTED_DECISION_SHA256,
    HOLDOUT_ANCHORS,
    SELECTED_FAMILY_ID,
    expand_holdout_cases,
    expand_holdout_methods,
    load_holdout_execution,
    validate_screen_decision,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_decision_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_execution_v1.json"
EXPECTED_EXECUTION_SHA256 = "b8a39f59247346e6a893e05d79d44082e64c6a38f3e0da618bf86f4195ccc2f1"


def synthetic_manifest() -> dict:
    cases = []
    for anchor_index in HOLDOUT_ANCHORS:
        for prompt_length in (1024, 2048, 4032):
            identity = {
                "anchor_index": anchor_index,
                "prompt_length": prompt_length,
                "continuation_start": anchor_index * 10000,
                "prompt_ids_sha256": f"p-{anchor_index}-{prompt_length}",
                "continuation_ids_sha256": f"c-{anchor_index}-{prompt_length}",
            }
            cases.append({
                "case_id": f"base-{anchor_index}-{prompt_length}",
                "identity": identity,
                "prompt_ids": [0] * prompt_length,
                "continuation_ids": [1] * 64,
            })
    return {"cases": cases}


class CageV3HoldoutTest(unittest.TestCase):
    def setUp(self):
        self.decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
        (
            self.execution,
            self.execution_sha256,
            self.protocol,
            _,
            self.plan,
            loaded_decision,
        ) = load_holdout_execution(
            EXECUTION_PATH,
            repo_root=REPO_ROOT,
            verify_artifacts=False,
        )
        self.assertEqual(loaded_decision, self.decision)

    def test_screen_decision_is_exactly_frozen_before_holdout(self):
        self.assertEqual(file_sha256(DECISION_PATH), EXPECTED_DECISION_SHA256)
        validate_screen_decision(
            self.decision,
            decision_path=DECISION_PATH,
            repo_root=REPO_ROOT,
            verify_artifacts=False,
        )
        self.assertEqual(self.decision["decision"]["selected_family_id"], SELECTED_FAMILY_ID)
        self.assertEqual(self.decision["decision"]["advanced_family_count"], 1)
        self.assertFalse(self.decision["consumption_state_at_freeze"]["holdout_metrics_consumed"])
        self.assertFalse(self.decision["consumption_state_at_freeze"]["reserved_unseen_metrics_consumed"])

    def test_execution_freezes_only_holdout_anchors_and_selected_family(self):
        self.assertEqual(file_sha256(EXECUTION_PATH), EXPECTED_EXECUTION_SHA256)
        self.assertEqual(self.execution_sha256, EXPECTED_EXECUTION_SHA256)
        self.assertEqual(self.execution["stages"]["holdout_acceptance"]["anchor_indices"], [10])
        self.assertEqual(self.execution["stages"]["holdout_full"]["anchor_indices"], HOLDOUT_ANCHORS)
        self.assertTrue(self.execution["authorization"]["holdout_metrics"])
        self.assertFalse(self.execution["authorization"]["reserved_unseen_metrics"])
        self.assertFalse(self.execution["authorization"]["end_to_end_quality"])

    def test_holdout_methods_include_selected_candidate_controls_and_kitty(self):
        cage = expand_holdout_methods(protocol=self.protocol, plan=self.plan, partition="cage_qwen3")
        kitty = expand_holdout_methods(protocol=self.protocol, plan=self.plan, partition="kitty_qwen3")
        self.assertEqual(len(cage), 6)
        self.assertEqual(len(kitty), 3)
        self.assertEqual([row["family_id"] for row in cage[:3]], ["cage-v2-best-control"] * 3)
        self.assertEqual([row["family_id"] for row in cage[3:]], [SELECTED_FAMILY_ID] * 3)
        self.assertEqual({row["family_id"] for row in kitty}, {"kitty-pro-25pct"})
        target_by_length = {row["prompt_length"]: row["packed_bytes"] for row in kitty}
        for method in cage[3:]:
            self.assertLessEqual(method["packed_bytes"], target_by_length[method["prompt_length"]])
            quotas = method["config"]["two_bit_channels"]
            self.assertEqual((quotas.count(48), quotas.count(32), quotas.count(16)), (12, 12, 12))
            self.assertEqual(sum(quotas), 1152)

    def test_expansion_has_exact_acceptance_and_full_grids(self):
        manifest = synthetic_manifest()
        expected = {
            ("cage_qwen3", "holdout_acceptance"): 6,
            ("cage_qwen3", "holdout_full"): 60,
            ("kitty_qwen3", "holdout_acceptance"): 3,
            ("kitty_qwen3", "holdout_full"): 30,
        }
        for (partition, stage), count in expected.items():
            cases = expand_holdout_cases(
                execution=self.execution,
                execution_sha256=self.execution_sha256,
                protocol=self.protocol,
                manifest=manifest,
                plan=self.plan,
                partition=partition,
                stage=stage,
            )
            self.assertEqual(len(cases), count)
            self.assertEqual(len({case["case_id"] for case in cases}), count)
            anchors = sorted({case["input"]["anchor_index"] for case in cases})
            self.assertEqual(anchors, [10] if stage == "holdout_acceptance" else HOLDOUT_ANCHORS)
            self.assertTrue(set(anchors).isdisjoint(range(20, 50)))

    def test_decision_mutations_cannot_expand_authorization(self):
        changed = copy.deepcopy(self.decision)
        changed["holdout_authorization"]["selected_candidate_family_count"] = 2
        with self.assertRaisesRegex(CageV3HoldoutError, "authorization"):
            validate_screen_decision(
                changed,
                decision_path=DECISION_PATH,
                repo_root=REPO_ROOT,
                verify_artifacts=False,
            )
        changed = copy.deepcopy(self.decision)
        changed["holdout_authorization"]["reserved_unseen_metrics"] = True
        with self.assertRaisesRegex(CageV3HoldoutError, "authorization"):
            validate_screen_decision(
                changed,
                decision_path=DECISION_PATH,
                repo_root=REPO_ROOT,
                verify_artifacts=False,
            )

    def test_acceptance_comparison_requires_bitwise_scientific_equality(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            identity = {"frozen": True}
            for root in (first, second):
                (root / "cases").mkdir(parents=True)
                summary = {
                    "status": "pass",
                    "stage": "holdout_acceptance",
                    "partition": "cage_qwen3",
                    "completed_cases": 6,
                    "new_cases": 6,
                    "resumed_cases": 0,
                    "failure_records": 0,
                    "identity": identity,
                }
                (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                for index in range(6):
                    record = {
                        "status": "completed",
                        "case_id": f"case-{index}",
                        "method": {"id": f"method-{index}"},
                        "input": {"anchor_index": 10},
                        "memory": {"model_total_bytes": index + 1},
                        "layer_metrics": [{"value": index}],
                        "aggregates": {"joint_post_o_proj_mse": {"mean": float(index)}},
                        "cache": {"ok": True},
                    }
                    (root / "cases" / f"case-{index}.json").write_text(json.dumps(record), encoding="utf-8")
            passed = compare_repeats(partition="cage_qwen3", first_root=first, second_root=second)
            self.assertEqual(passed["status"], "pass")
            self.assertEqual(passed["mismatch_case_ids"], [])
            changed_path = second / "cases" / "case-3.json"
            changed = json.loads(changed_path.read_text(encoding="utf-8"))
            changed["aggregates"]["joint_post_o_proj_mse"]["mean"] += 1e-9
            changed_path.write_text(json.dumps(changed), encoding="utf-8")
            failed = compare_repeats(partition="cage_qwen3", first_root=first, second_root=second)
            self.assertEqual(failed["status"], "fail")
            self.assertEqual(failed["mismatch_case_ids"], ["case-3"])


if __name__ == "__main__":
    unittest.main()
