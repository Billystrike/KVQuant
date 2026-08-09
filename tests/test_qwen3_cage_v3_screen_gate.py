import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_acceptance_gate_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_execution_v1.json"
EXPECTED_GATE_SHA256 = "89329379bed30bb82b7974252bc453156b27f503d39331a80de4a0d523b09d7f"
EXPECTED_SOURCE_SHA256 = {
    "runner": "890410f6058f05ec6f208a81ed961fbf8dc69e829ae0dca08415d2e4e3921799",
    "comparator": "85460ca3fbf73d7cb0d9ffa4cdbe01dd3d0f3fcc348ad9aea68752bfa49f2b28",
    "screen_utils": "cc7bde456379e2d13ec5d90019426f0dd64d30bea2626fd7426e6000554accf6",
    "round1_runtime": "3dc1528a2cfff9a84f7e0bd8cd1f1d740c6e264295803bcfdfba8767344010b8",
    "qwen3_cage": "d9aff2e539961b118d2b9e8375f5ed615d1fe4fa18ea929250bda8b617a2d439",
    "qwen3_cage_v2": "a183da805e9acec7ab8af3c733a447bf27ba42d26a13377f86718550553a7367",
    "cage_v2_quant": "2fe57d20ccc34506789254183f3da1f82c017b9e0882f0f7c530c589b9569a8a",
    "cage_v2_memory": "aa5f6e9259d3eaf635e07614241c0d0aa1e95cac4d9bc342079dde80679ef0d0",
    "recorder": "f699776de29f779379c4cfc2f6b65c9644503cfa8231606c717b8ff4d7e6ac82",
}


class CageV3ScreenGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))

    def test_gate_freezes_both_partition_repeats_and_sources(self):
        self.assertEqual(file_sha256(GATE_PATH), EXPECTED_GATE_SHA256)
        self.assertEqual(self.gate["schema_version"], 1)
        self.assertEqual(self.gate["status"], "pass")
        self.assertFalse(self.gate["claim_eligible"])
        self.assertEqual(self.gate["execution_sha256"], file_sha256(EXECUTION_PATH))
        self.assertEqual(
            self.gate["acceptance_source"]["git_commit"],
            "a8917a1d04101470cfcf2007f704f55a7a55b8e8",
        )
        self.assertEqual(self.gate["acceptance_source"]["source_sha256"], EXPECTED_SOURCE_SHA256)
        expected = {
            "cage_qwen3": (12, "69f7bac3a3ed2326d4b74792c2f80e48801a3cded08db9bbbb56c7b64552d7b9"),
            "kitty_qwen3": (3, "7de6eec9d83f0488d738e2666083e90a90da8a0542c1cd18ec6d5f8303c09684"),
        }
        for partition, (count, payload) in expected.items():
            record = self.gate["partitions"][partition]
            self.assertEqual(record["case_count"], count)
            self.assertEqual(record["scientific_payload_sha256"], payload)
            for repeat in ("repeat_a", "repeat_b"):
                self.assertEqual(record[repeat]["execution_log_size_bytes"], 10281 if partition == "cage_qwen3" else 8388)

    def test_gate_only_authorizes_registered_screen_partition(self):
        authorization = self.gate["screen_full_authorization"]
        self.assertEqual(authorization["anchor_indices"], list(range(5, 10)))
        self.assertEqual(authorization["cage_case_count"], 60)
        self.assertEqual(authorization["kitty_case_count"], 15)
        self.assertFalse(authorization["holdout_metrics"])
        self.assertFalse(authorization["reserved_unseen_metrics"])


if __name__ == "__main__":
    unittest.main()
