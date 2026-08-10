import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_holdout_analysis import (
    CageV3HoldoutAnalysisError,
    EXPECTED_HOLDOUT_RECEIPT_SHA256,
    validate_holdout_receipt,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_postrun_receipt_v1.json"


class CageV3HoldoutReceiptTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_receipt_freezes_passed_holdout_before_interpretation(self):
        self.assertEqual(file_sha256(RECEIPT_PATH), EXPECTED_HOLDOUT_RECEIPT_SHA256)
        validate_holdout_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        self.assertEqual(
            self.receipt["audit_summary"]["joint_scientific_payload_sha256"],
            "fc2ae7f310adf98251900d235c5e21c6769131e38b89641679c9535a4e80d5f2",
        )
        self.assertFalse(self.receipt["audit_summary"]["reserved_unseen_metrics_consumed"])

    def test_receipt_rejects_interpretation_or_expanded_authorization(self):
        changed = copy.deepcopy(self.receipt)
        changed["interpretation_performed"] = True
        with self.assertRaisesRegex(CageV3HoldoutAnalysisError, "interpreted"):
            validate_holdout_receipt(changed, receipt_path=RECEIPT_PATH)
        changed = copy.deepcopy(self.receipt)
        changed["next_authorization"]["reserved_unseen_metrics"] = True
        with self.assertRaisesRegex(CageV3HoldoutAnalysisError, "authorization"):
            validate_holdout_receipt(changed, receipt_path=RECEIPT_PATH)


if __name__ == "__main__":
    unittest.main()
