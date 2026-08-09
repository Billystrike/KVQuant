import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_postrun_receipt_v1.json"
EXPECTED_RECEIPT_SHA256 = "c8cb023245171919ba9a7b75dd809cbd963661d6117f206366b15276baa7b277"


def validate_receipt(receipt):
    if receipt.get("schema_version") != 1:
        raise ValueError("receipt schema mismatch")
    if receipt.get("status") != "frozen_after_joint_postrun_audit_before_screen_interpretation":
        raise ValueError("receipt status mismatch")
    if receipt.get("claim_eligible") is not False or receipt.get("interpretation_performed") is not False:
        raise ValueError("receipt interpretation boundary mismatch")
    summary = receipt.get("audit_summary", {})
    if (summary.get("status"), summary.get("case_count"), summary.get("layer_record_count"), summary.get("failure_count")) != ("pass", 75, 2700, 0):
        raise ValueError("receipt audit totals mismatch")
    if summary.get("holdout_metrics_consumed") is not False or summary.get("reserved_unseen_metrics_consumed") is not False:
        raise ValueError("receipt partition boundary mismatch")
    partitions = receipt.get("partitions", {})
    if tuple(partitions) != ("cage_qwen3", "kitty_qwen3"):
        raise ValueError("receipt partitions mismatch")
    if partitions["cage_qwen3"].get("case_count") != 60 or partitions["kitty_qwen3"].get("case_count") != 15:
        raise ValueError("receipt partition counts mismatch")
    authorization = receipt.get("next_authorization", {})
    if authorization != {
        "screen_interpretation": True,
        "maximum_candidates_advanced": 1,
        "holdout_metrics": False,
        "reserved_unseen_metrics": False,
        "end_to_end_quality": False,
    }:
        raise ValueError("receipt authorization mismatch")


class CageV3ScreenReceiptTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_receipt_freezes_passed_audit_before_interpretation(self):
        self.assertEqual(file_sha256(RECEIPT_PATH), EXPECTED_RECEIPT_SHA256)
        validate_receipt(self.receipt)
        self.assertEqual(self.receipt["audit"]["sha256"], "bf0601bd338f3efb1282fbcf46443b3804dac1ef259f7829a1ca87424cdc9484")
        self.assertEqual(self.receipt["audit_summary"]["joint_scientific_payload_sha256"], "3b4c93981550f81a55324c9682f87f81842185a8b26fddd81681d55ec19b8678")

    def test_receipt_rejects_interpretation_or_holdout_authorization_mutation(self):
        changed = copy.deepcopy(self.receipt)
        changed["interpretation_performed"] = True
        with self.assertRaisesRegex(ValueError, "interpretation"):
            validate_receipt(changed)
        changed = copy.deepcopy(self.receipt)
        changed["next_authorization"]["holdout_metrics"] = True
        with self.assertRaisesRegex(ValueError, "authorization"):
            validate_receipt(changed)


if __name__ == "__main__":
    unittest.main()
