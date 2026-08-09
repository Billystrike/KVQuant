import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_calibration import (
    CageV3CalibrationPostrunError,
    FAMILIES,
    PROMPT_LENGTHS,
    derive_layer_quota_plans,
    shell_case_manifest_sha256,
    validate_receipt,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibration_full_receipt_v1.json"
PLAN_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibrated_quota_plan_v1.json"
PROTOCOL_SHA256 = "c92e452b5eac99c7a015da21a82080cf61ddd070e677cc3664ed78bf657cbfe9"
EXECUTION_SHA256 = "f97a25e85a4bc82a64c74c877309dc5e38f3ee5b908a6ccf42c7aaee108d2fb6"
EXPECTED_RECEIPT_SHA256 = "f6d7d57d683881a572b7c1eaf72e810891bb30f2f71f16b3c9f067c11e9562c6"
EXPECTED_PLAN_SHA256 = "01a1063651d4565a732474b9f27f7a674f434750e7aea2f55429f619bed91914"


class CageV3CalibrationPostrunTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_checked_in_receipt_freezes_only_calibration_artifacts(self):
        self.assertEqual(file_sha256(RECEIPT_PATH), EXPECTED_RECEIPT_SHA256)
        validate_receipt(
            self.receipt,
            repo_root=REPO_ROOT,
            protocol_sha256=PROTOCOL_SHA256,
            execution_sha256=EXECUTION_SHA256,
        )
        self.assertEqual(self.receipt["calibration_full"]["case_count"], 30)
        self.assertEqual(self.receipt["calibration_full"]["layer_record_count"], 1080)
        authorization = self.receipt["derivation_authorization"]
        self.assertFalse(authorization["screen_metrics"])
        self.assertFalse(authorization["holdout_metrics"])
        self.assertFalse(authorization["reserved_unseen_metrics"])

    def test_receipt_rejects_posthoc_partition_authorization(self):
        changed = copy.deepcopy(self.receipt)
        changed["derivation_authorization"]["screen_metrics"] = True
        with self.assertRaisesRegex(CageV3CalibrationPostrunError, "screen_metrics"):
            validate_receipt(
                changed,
                repo_root=REPO_ROOT,
                protocol_sha256=PROTOCOL_SHA256,
                execution_sha256=EXECUTION_SHA256,
            )

    def test_shell_manifest_uses_frozen_relative_case_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "b.json"
            second = root / "a.json"
            first.write_text("b", encoding="utf-8")
            second.write_text("a", encoding="utf-8")
            digest = shell_case_manifest_sha256([first, second])
            import hashlib

            expected = hashlib.sha256(
                (
                    f"{file_sha256(second)}  cases/a.json\n"
                    f"{file_sha256(first)}  cases/b.json\n"
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(digest, expected)

    def test_plan_grid_is_exactly_six_registered_groups(self):
        self.assertEqual(
            [(family, length) for family in FAMILIES for length in PROMPT_LENGTHS],
            [
                ("pure-sr2-sink32-uniform32", 1024),
                ("pure-sr2-sink32-uniform32", 2048),
                ("pure-sr2-sink32-uniform32", 4032),
                ("pure-sr2-sink64-uniform32", 1024),
                ("pure-sr2-sink64-uniform32", 2048),
                ("pure-sr2-sink64-uniform32", 4032),
            ],
        )

    def test_synthetic_quota_derivation_obeys_registered_ranking_and_tiers(self):
        records = []
        for family_id in FAMILIES:
            for prompt_length in PROMPT_LENGTHS:
                for anchor_index in range(5):
                    records.append({
                        "method": {"family_id": family_id},
                        "input": {
                            "prompt_length": prompt_length,
                            "anchor_index": anchor_index,
                        },
                        "layer_metrics": [
                            {
                                "layer_idx": layer_idx,
                                "metrics": {
                                    "joint_post_o_proj_mse": float(layer_idx + anchor_index)
                                },
                            }
                            for layer_idx in range(36)
                        ],
                    })
        plans = derive_layer_quota_plans(records)
        self.assertEqual(len(plans), 6)
        for plan in plans:
            self.assertEqual(plan["ranked_layer_indices"], list(reversed(range(36))))
            self.assertEqual(plan["layer_two_bit_channel_quotas"][:12], [16] * 12)
            self.assertEqual(plan["layer_two_bit_channel_quotas"][12:24], [32] * 12)
            self.assertEqual(plan["layer_two_bit_channel_quotas"][24:], [48] * 12)
            self.assertEqual(plan["quota_total"], 1152)

    def test_checked_in_quota_plan_is_exact_server_artifact(self):
        self.assertEqual(file_sha256(PLAN_PATH), EXPECTED_PLAN_SHA256)
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(plan["receipt_sha256"], EXPECTED_RECEIPT_SHA256)
        self.assertEqual(
            plan["calibration_scientific_payload_sha256"],
            "949e49b0039b7084a9fc3cd1133036e38c86ce4bf5b9bcebdc77bd6adb2b1fac",
        )
        self.assertFalse(plan["screen_metrics_consumed"])
        self.assertFalse(plan["holdout_metrics_consumed"])
        self.assertFalse(plan["reserved_unseen_metrics_consumed"])
        self.assertEqual(len(plan["plans"]), 6)
        for record in plan["plans"]:
            scores = record["layer_scores"]
            ranking = sorted(range(36), key=lambda layer_idx: (-scores[layer_idx], layer_idx))
            self.assertEqual(record["ranked_layer_indices"], ranking)
            quotas = [32] * 36
            for layer_idx in ranking[:12]:
                quotas[layer_idx] = 48
            for layer_idx in ranking[-12:]:
                quotas[layer_idx] = 16
            self.assertEqual(record["layer_two_bit_channel_quotas"], quotas)
            self.assertEqual(record["quota_counts"], {"16": 12, "32": 12, "48": 12})
            self.assertEqual(record["quota_total"], 1152)


if __name__ == "__main__":
    unittest.main()
