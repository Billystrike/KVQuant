import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibration_acceptance_gate_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibration_execution_v1.json"
EXPECTED_GATE_SHA256 = "b3961d72f529a8ebf6c949a39a13f80a078700b8b8ec92044b21e10f9ac29a44"
EXPECTED_SOURCE_SHA256 = {
    "runner": "81768a00d9c148f93e77f25cd41032d727f70e3aee9d6c32b062d2383c2806e1",
    "comparator": "fac39e675f01aa3ac4cdf4712e11b546aeffdbd67030ce4341465445e41fa1d1",
    "execution_utils": "e6622e3656d86c8a6e3ef6c4a7a0c62827a68debd30bcb4e23b16d971dfb4dcb",
    "protocol_utils": "88e44e4feddc2eba4817b0753943cfeb5bd0ca1c7933f1ec5dfd9edcca45d6a7",
    "qwen3_cage": "d9aff2e539961b118d2b9e8375f5ed615d1fe4fa18ea929250bda8b617a2d439",
    "qwen3_cage_v2": "a183da805e9acec7ab8af3c733a447bf27ba42d26a13377f86718550553a7367",
    "cage_v2_quant": "2fe57d20ccc34506789254183f3da1f82c017b9e0882f0f7c530c589b9569a8a",
    "cage_v2_memory": "aa5f6e9259d3eaf635e07614241c0d0aa1e95cac4d9bc342079dde80679ef0d0",
    "recorder": "f699776de29f779379c4cfc2f6b65c9644503cfa8231606c717b8ff4d7e6ac82",
}


class CageV3CalibrationGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        self.execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))

    def test_gate_freezes_final_repeat_and_scientific_sources(self):
        self.assertEqual(file_sha256(GATE_PATH), EXPECTED_GATE_SHA256)
        self.assertEqual(self.gate["schema_version"], 1)
        self.assertEqual(
            self.gate["gate_id"],
            "qwen3-8b-cage-v3-calibration-acceptance-gate-v1",
        )
        self.assertEqual(self.gate["status"], "pass")
        self.assertFalse(self.gate["claim_eligible"])
        self.assertEqual(
            self.gate["acceptance_source"]["git_commit"],
            "313a8ff43eb9a9504bdbe12b9f68ab6b0e54b919",
        )
        self.assertEqual(
            self.gate["acceptance_source"]["source_sha256"],
            EXPECTED_SOURCE_SHA256,
        )
        for name in ("repeat_a", "repeat_b"):
            self.assertIn("313a8ff", self.gate[name]["output_dir"])
            self.assertEqual(self.gate[name]["execution_log_size_bytes"], 4791)
        self.assertEqual(
            self.gate["comparison"]["scientific_payload_sha256"],
            "a7eac3563cdcf9155f1078d1de06fe60b1de9aac544646fb93bfec019d68717a",
        )

    def test_gate_matches_frozen_execution_and_only_authorizes_calibration(self):
        self.assertEqual(self.gate["execution_sha256"], file_sha256(EXECUTION_PATH))
        self.assertEqual(
            self.gate["protocol_sha256"], self.execution["protocol"]["sha256"]
        )
        self.assertEqual(
            self.gate["manifest_sha256"], self.execution["input_manifest"]["sha256"]
        )
        authorization = self.gate["calibration_full_authorization"]
        self.assertEqual(authorization["anchor_indices"], list(range(5)))
        self.assertEqual(authorization["case_count"], 30)
        self.assertFalse(authorization["screen_metrics"])
        self.assertFalse(authorization["holdout_metrics"])
        self.assertFalse(authorization["reserved_unseen_metrics"])

    def test_gate_mutations_break_frozen_identity(self):
        changed = copy.deepcopy(self.gate)
        changed["acceptance_source"]["source_sha256"]["runner"] = "0" * 64
        self.assertNotEqual(
            changed["acceptance_source"]["source_sha256"], EXPECTED_SOURCE_SHA256
        )
        changed = copy.deepcopy(self.gate)
        changed["calibration_full_authorization"]["screen_metrics"] = True
        self.assertNotEqual(
            changed["calibration_full_authorization"],
            {
                "anchor_indices": list(range(5)),
                "case_count": 30,
                "screen_metrics": False,
                "holdout_metrics": False,
                "reserved_unseen_metrics": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
