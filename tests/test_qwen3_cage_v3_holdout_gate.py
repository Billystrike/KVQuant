import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_acceptance_gate_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_execution_v1.json"
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_decision_v1.json"
EXPECTED_GATE_SHA256 = "e83cdccdb1faf9005ab7a5cbdbecd3c63ede04e27ed9b94862e718a5eb7cd576"
EXPECTED_SOURCE_SHA256 = {
    "runner": "bcb283a064d719fa816de2de5896238e16d44b3adb881a6eaa32164d2c429035",
    "comparator": "390e9644487fb7e0199399fd7566c6df76c712a60b2de36dd49f76c292512f60",
    "holdout_utils": "5fc61df7095a5dc2442d22e50b0992596e343209c0383e52b716c236afa0aed3",
    "round1_runtime": "3dc1528a2cfff9a84f7e0bd8cd1f1d740c6e264295803bcfdfba8767344010b8",
    "qwen3_cage": "d9aff2e539961b118d2b9e8375f5ed615d1fe4fa18ea929250bda8b617a2d439",
    "qwen3_cage_v2": "a183da805e9acec7ab8af3c733a447bf27ba42d26a13377f86718550553a7367",
    "cage_v2_quant": "2fe57d20ccc34506789254183f3da1f82c017b9e0882f0f7c530c589b9569a8a",
    "cage_v2_memory": "aa5f6e9259d3eaf635e07614241c0d0aa1e95cac4d9bc342079dde80679ef0d0",
    "recorder": "f699776de29f779379c4cfc2f6b65c9644503cfa8231606c717b8ff4d7e6ac82",
}


def current_source_hashes():
    return {
        "runner": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v3_holdout.py"),
        "comparator": file_sha256(REPO_ROOT / "scripts" / "qwen3_compare_cage_v3_holdout_acceptance.py"),
        "holdout_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_holdout.py"),
        "round1_runtime": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v2_round1.py"),
        "qwen3_cage": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
        "qwen3_cage_v2": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
        "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
        "cage_v2_memory": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v2.py"),
        "recorder": file_sha256(REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"),
    }


class CageV3HoldoutGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))

    def test_gate_freezes_both_partition_repeats_and_accepted_sources(self):
        self.assertEqual(file_sha256(GATE_PATH), EXPECTED_GATE_SHA256)
        self.assertEqual(self.gate["schema_version"], 1)
        self.assertEqual(self.gate["status"], "pass")
        self.assertFalse(self.gate["claim_eligible"])
        self.assertEqual(self.gate["execution_sha256"], file_sha256(EXECUTION_PATH))
        self.assertEqual(self.gate["screen_decision_sha256"], file_sha256(DECISION_PATH))
        self.assertEqual(
            self.gate["acceptance_source"]["git_commit"],
            "ba3130b8305a3c9ff8d70255bd44e82af331ec1b",
        )
        self.assertEqual(self.gate["acceptance_source"]["source_sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(current_source_hashes(), EXPECTED_SOURCE_SHA256)
        expected = {
            "cage_qwen3": (
                6,
                "371b758bc5602b324285e8185b367b9c4f24e694fb554039c450e8f346a02e18",
                "59dbf66ddbfcd5dcf780aa12ea9df1ad673afd74a6f53026541c345c6d2cc349",
            ),
            "kitty_qwen3": (
                3,
                "2826ddee6c7327b262ea23c13267eb4ded66f27921939615dc88a766cd166863",
                "4e2d8ce8a0615e11fb4be5c1de8c14b0292fd87b1c2fa156494537c70488e49c",
            ),
        }
        for partition, (count, payload, comparison) in expected.items():
            record = self.gate["partitions"][partition]
            self.assertEqual(record["case_count"], count)
            self.assertEqual(record["scientific_payload_sha256"], payload)
            self.assertEqual(record["comparison"]["sha256"], comparison)
            self.assertNotEqual(
                record["repeat_a"]["case_file_manifest_sha256"],
                record["repeat_b"]["case_file_manifest_sha256"],
            )
            self.assertEqual(
                record["repeat_a"]["run_identity_sha256"],
                record["repeat_b"]["run_identity_sha256"],
            )
            self.assertEqual(
                record["repeat_a"]["summary_sha256"],
                record["repeat_b"]["summary_sha256"],
            )

    def test_gate_authorizes_only_registered_full_holdout(self):
        authorization = self.gate["holdout_full_authorization"]
        self.assertEqual(authorization["anchor_indices"], list(range(10, 20)))
        self.assertEqual(authorization["cage_case_count"], 60)
        self.assertEqual(authorization["kitty_case_count"], 30)
        self.assertEqual(authorization["selected_family_id"], "pure-sr2-sink32-calibrated")
        self.assertFalse(authorization["reserved_unseen_metrics"])
        self.assertFalse(authorization["end_to_end_quality"])


if __name__ == "__main__":
    unittest.main()
