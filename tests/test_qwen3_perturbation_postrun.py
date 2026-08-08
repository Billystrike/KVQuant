import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


from scripts.qwen3_validate_perturbation_full import (
    _case_manifest,
    _validate_artifact_manifest,
)
from utils.qwen3_perturbation_protocol import Qwen3PerturbationError


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPO_ROOT / "configs" / "qwen3_8b_memory_perturbation_full_artifacts_v1.json"
)
PROTOCOL_SHA = "90850f59347393c5c164c1f84737a1fb258db9cfb4ee0126bc4517eea9819e12"
GATE_SHA = "ba4229bf54a1d7ea4998fa025edefde6191477f840eca9a90f2f610567a25fe1"


class Qwen3PerturbationPostrunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    def test_full_artifact_manifest_freezes_both_partition_receipts(self):
        _validate_artifact_manifest(
            self.artifacts,
            protocol_sha256=PROTOCOL_SHA,
            gate_sha256=GATE_SHA,
        )
        self.assertEqual(self.artifacts["partitions"]["cage_qwen3"]["expected_cases"], 1000)
        self.assertEqual(self.artifacts["partitions"]["kitty_qwen3"]["expected_cases"], 300)
        self.assertFalse(
            self.artifacts["joint_expected"]["interpretation_allowed_before_audit_pass"]
        )

    def test_full_artifact_manifest_rejects_result_hash_mutation(self):
        mutated = copy.deepcopy(self.artifacts)
        mutated["acceptance_gate_sha256"] = "0" * 64
        with self.assertRaises(Qwen3PerturbationError):
            _validate_artifact_manifest(
                mutated,
                protocol_sha256=PROTOCOL_SHA,
                gate_sha256=GATE_SHA,
            )

    def test_case_manifest_is_path_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text('{"a":1}\n', encoding="utf-8")
            second.write_text('{"b":2}\n', encoding="utf-8")
            canonical, shell = _case_manifest([first, second])
            expected_records = [
                {
                    "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in (first, second)
            ]
            expected_canonical = hashlib.sha256(
                json.dumps(
                    expected_records,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(canonical, expected_canonical)
            self.assertEqual(len(shell), 64)


if __name__ == "__main__":
    unittest.main()
