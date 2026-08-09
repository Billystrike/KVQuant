import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import (
    CageV3ScreenError,
    expand_screen_cases,
    expand_screen_methods,
    load_screen_execution,
    validate_quota_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_execution_v1.json"
EXPECTED_EXECUTION_SHA256 = "2f94fdb35c1c087270b971b524cb9a26722676d1d950fa8588037027a90f7dc0"


class CageV3ScreenTest(unittest.TestCase):
    def setUp(self):
        (
            self.execution,
            self.execution_sha256,
            self.protocol,
            _,
            self.plan,
        ) = load_screen_execution(
            EXECUTION_PATH,
            repo_root=REPO_ROOT,
            verify_artifacts=False,
        )

    def test_execution_and_plan_hashes_are_frozen_before_screen(self):
        self.assertEqual(file_sha256(EXECUTION_PATH), EXPECTED_EXECUTION_SHA256)
        self.assertEqual(self.execution_sha256, EXPECTED_EXECUTION_SHA256)
        self.assertEqual(
            self.execution["quota_plan"]["sha256"],
            "01a1063651d4565a732474b9f27f7a674f434750e7aea2f55429f619bed91914",
        )
        self.assertEqual(self.execution["stages"]["screen_acceptance"]["anchor_indices"], [5])
        self.assertEqual(self.execution["stages"]["screen_full"]["anchor_indices"], list(range(5, 10)))
        self.assertFalse(self.execution["authorization"]["holdout_metrics"])
        self.assertFalse(self.execution["authorization"]["reserved_unseen_metrics"])

    def test_methods_bind_controls_candidates_kitty_and_calibrated_quotas(self):
        cage = expand_screen_methods(
            protocol=self.protocol,
            plan=self.plan,
            partition="cage_qwen3",
        )
        kitty = expand_screen_methods(
            protocol=self.protocol,
            plan=self.plan,
            partition="kitty_qwen3",
        )
        self.assertEqual(len(cage), 12)
        self.assertEqual(len(kitty), 3)
        self.assertEqual([row["family_id"] for row in cage[:3]], ["cage-v2-best-control"] * 3)
        self.assertEqual({row["config"]["boosted_channels"] for row in kitty}, {32})
        calibrated = [row for row in cage if "calibrated" in row["family_id"]]
        self.assertEqual(len(calibrated), 6)
        for method in calibrated:
            quotas = method["config"]["two_bit_channels"]
            self.assertEqual(len(quotas), 36)
            self.assertEqual((quotas.count(48), quotas.count(32), quotas.count(16)), (12, 12, 12))
            self.assertEqual(sum(quotas), 1152)
            self.assertLessEqual(method["packed_bytes"], method["target_bytes"])

    def test_expanded_acceptance_and_full_counts_are_exact(self):
        manifest = {"cases": []}
        for anchor_index in range(5, 10):
            for prompt_length in (1024, 2048, 4032):
                identity = {
                    "anchor_index": anchor_index,
                    "prompt_length": prompt_length,
                    "continuation_start": anchor_index * 10000,
                    "prompt_ids_sha256": f"p-{anchor_index}-{prompt_length}",
                    "continuation_ids_sha256": f"c-{anchor_index}-{prompt_length}",
                }
                manifest["cases"].append({
                    "case_id": f"base-{anchor_index}-{prompt_length}",
                    "identity": identity,
                    "prompt_ids": list(range(prompt_length)),
                    "continuation_ids": [1] * 64,
                })
        expected = {
            ("cage_qwen3", "screen_acceptance"): 12,
            ("cage_qwen3", "screen_full"): 60,
            ("kitty_qwen3", "screen_acceptance"): 3,
            ("kitty_qwen3", "screen_full"): 15,
        }
        for (partition, stage), count in expected.items():
            cases = expand_screen_cases(
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
            allowed = [5] if stage == "screen_acceptance" else list(range(5, 10))
            self.assertEqual(sorted({case["input"]["anchor_index"] for case in cases}), allowed)

    def test_quota_plan_rejects_screen_or_holdout_contamination(self):
        changed = copy.deepcopy(self.plan)
        changed["screen_metrics_consumed"] = True
        with self.assertRaisesRegex(CageV3ScreenError, "screen_metrics_consumed"):
            validate_quota_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed["plans"][0]["layer_two_bit_channel_quotas"][0] = 48
        with self.assertRaisesRegex(CageV3ScreenError, "allocation"):
            validate_quota_plan(changed)


if __name__ == "__main__":
    unittest.main()
