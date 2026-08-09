import copy
import hashlib
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v2_gate import (
    CageV2AcceptanceGateError,
    load_cage_v2_acceptance_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v2_round1_acceptance_gate_v1.json"
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v2_feasibility_protocol_draft_v1.json"


class CageV2AcceptanceGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        self.protocol_sha256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
        self.manifest_sha256 = self.gate["development_manifest"]["sha256"]

    def _validate_static_copy(self, gate):
        from unittest.mock import patch

        manifest_path = Path(self.gate["development_manifest"]["path"])
        hashes = {
            GATE_PATH: hashlib.sha256(
                json.dumps(gate, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            PROTOCOL_PATH: self.protocol_sha256,
            manifest_path: self.manifest_sha256,
        }
        with patch("utils.qwen3_cage_v2_gate.load_json", return_value=gate), patch(
            "utils.qwen3_cage_v2_gate.file_sha256",
            side_effect=lambda path: hashes[Path(path)],
        ):
            return load_cage_v2_acceptance_gate(
                GATE_PATH,
                protocol_path=PROTOCOL_PATH,
                manifest_path=manifest_path,
                verify_artifacts=False,
            )

    def test_checked_in_gate_freezes_both_repeat_pairs_and_screen_count(self):
        gate, _ = self._validate_static_copy(self.gate)
        self.assertFalse(gate["claim_eligible"])
        self.assertEqual(gate["partitions"]["cage_qwen3"]["case_count_per_repeat"], 23)
        self.assertEqual(gate["partitions"]["kitty_qwen3"]["case_count_per_repeat"], 6)
        self.assertEqual(gate["round1_screen_authorization"]["total_cases"], 145)
        self.assertTrue(gate["round1_screen_authorization"]["requires_exact_execution_commit"])

    def test_gate_rejects_protocol_or_screen_count_mutation(self):
        mutated = copy.deepcopy(self.gate)
        mutated["protocol"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CageV2AcceptanceGateError, "protocol hash"):
            self._validate_static_copy(mutated)

        mutated = copy.deepcopy(self.gate)
        mutated["round1_screen_authorization"]["total_cases"] += 1
        with self.assertRaisesRegex(CageV2AcceptanceGateError, "case total"):
            self._validate_static_copy(mutated)


if __name__ == "__main__":
    unittest.main()
