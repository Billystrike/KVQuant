import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_analysis import EXPECTED_RECEIPT_SHA256, validate_screen_receipt
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_postrun_receipt_v1.json"


class CageV3ScreenReceiptTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_receipt_freezes_passed_audit_before_interpretation(self):
        self.assertEqual(file_sha256(RECEIPT_PATH), EXPECTED_RECEIPT_SHA256)
        validate_screen_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        self.assertEqual(self.receipt["audit"]["sha256"], "bf0601bd338f3efb1282fbcf46443b3804dac1ef259f7829a1ca87424cdc9484")
        self.assertEqual(self.receipt["audit_summary"]["joint_scientific_payload_sha256"], "3b4c93981550f81a55324c9682f87f81842185a8b26fddd81681d55ec19b8678")

if __name__ == "__main__":
    unittest.main()
