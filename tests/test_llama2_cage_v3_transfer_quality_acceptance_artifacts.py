import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_data import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
A_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_a_artifacts_v1.json"
B_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_b_artifacts_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_gate_receipt_v2.json"


class Llama2CageV3TransferQualityAcceptanceArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repeat_a = json.loads(A_PATH.read_text(encoding="utf-8"))
        cls.repeat_b = json.loads(B_PATH.read_text(encoding="utf-8"))

    def test_two_repeat_artifacts_are_frozen_for_comparison(self):
        self.assertEqual(self.repeat_a["status"], "pass")
        self.assertEqual(self.repeat_b["status"], "pass")
        self.assertFalse(self.repeat_a["claim_eligible"])
        self.assertFalse(self.repeat_b["claim_eligible"])
        self.assertEqual(self.repeat_a["repeat"], "a")
        self.assertEqual(self.repeat_b["repeat"], "b")
        self.assertEqual(self.repeat_a["output"]["case_count"], 12)
        self.assertEqual(self.repeat_b["output"]["case_count"], 12)
        self.assertEqual(self.repeat_a["output"]["failure_count"], 0)
        self.assertEqual(self.repeat_b["output"]["failure_count"], 0)
        self.assertEqual(
            self.repeat_a["output"]["scientific_payload_sha256"],
            "bf0dcb6deb38bc1b6c7433f911b8abcc0ed819b92dac32ffe9c70f61b285b592",
        )
        self.assertEqual(
            self.repeat_b["output"]["scientific_payload_sha256"],
            self.repeat_a["output"]["scientific_payload_sha256"],
        )

    def test_gate_and_repeat_a_links_are_exact(self):
        self.assertEqual(self.repeat_b["gate_receipt"]["sha256"], file_sha256(GATE_PATH))
        self.assertEqual(
            self.repeat_b["repeat_a_artifacts"]["scientific_payload_sha256"],
            self.repeat_a["output"]["scientific_payload_sha256"],
        )

    def test_artifacts_do_not_prematurely_authorize_formal_execution(self):
        self.assertTrue(self.repeat_b["decision"]["acceptance_comparison_authorized"])
        self.assertFalse(self.repeat_b["decision"]["acceptance_consistency_established"])
        for key in (
            "full_600_case_execution_authorized",
            "candidate_tuning_authorized",
            "paper_claims_authorized",
            "runtime_claims_authorized",
        ):
            self.assertFalse(self.repeat_a["decision"][key])
            self.assertFalse(self.repeat_b["decision"][key])


if __name__ == "__main__":
    unittest.main()
