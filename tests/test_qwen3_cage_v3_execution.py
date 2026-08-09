import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_execution import (
    SCIENTIFIC_FIELDS,
    expand_calibration_cases,
    load_calibration_execution,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibration_execution_v1.json"
EXPECTED_EXECUTION_SHA256 = "f97a25e85a4bc82a64c74c877309dc5e38f3ee5b908a6ccf42c7aaee108d2fb6"


class CageV3ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.execution, self.execution_sha256, self.protocol, _ = load_calibration_execution(
            EXECUTION_PATH,
            repo_root=REPO_ROOT,
            verify_artifacts=False,
        )

    def _synthetic_manifest(self):
        cases = []
        for anchor in range(5):
            for length in (1024, 2048, 4032):
                cases.append({
                    "case_id": f"base-{anchor}-{length}",
                    "identity": {
                        "anchor_index": anchor,
                        "prompt_length": length,
                        "continuation_start": 100000 + anchor * 10000,
                        "prompt_ids_sha256": f"prompt-{anchor}-{length}",
                        "continuation_ids_sha256": f"continuation-{anchor}",
                    },
                    "prompt_ids": [1] * length,
                    "continuation_ids": [2] * 64,
                })
        return {"cases": cases}

    def test_checked_in_execution_hash_and_authorization_are_frozen(self):
        self.assertEqual(file_sha256(EXECUTION_PATH), EXPECTED_EXECUTION_SHA256)
        self.assertEqual(self.execution_sha256, EXPECTED_EXECUTION_SHA256)
        self.assertEqual(tuple(self.execution["scientific_payload_fields"]), SCIENTIFIC_FIELDS)
        self.assertTrue(self.execution["authorization"]["calibration_metrics"])
        self.assertFalse(self.execution["authorization"]["screen_metrics"])
        self.assertFalse(self.execution["authorization"]["holdout_metrics"])
        self.assertFalse(self.execution["authorization"]["reserved_unseen_metrics"])

    def test_expansion_freezes_six_acceptance_and_thirty_full_cases(self):
        manifest = self._synthetic_manifest()
        acceptance = expand_calibration_cases(
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            protocol=self.protocol,
            manifest=manifest,
            stage="calibration_acceptance",
        )
        full = expand_calibration_cases(
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            protocol=self.protocol,
            manifest=manifest,
            stage="calibration_full",
        )
        self.assertEqual(len(acceptance), 6)
        self.assertEqual(len(full), 30)
        self.assertEqual({case["input"]["anchor_index"] for case in acceptance}, {0})
        self.assertEqual({case["input"]["anchor_index"] for case in full}, set(range(5)))
        self.assertTrue(all(case["method"]["config"]["one_bit_channels"] == 0 for case in full))
        self.assertTrue(all(case["method"]["config"]["two_bit_channels"] == 32 for case in full))
        self.assertEqual(len({case["case_id"] for case in full}), 30)

    def test_execution_rejects_posthoc_screen_authorization(self):
        changed = copy.deepcopy(self.execution)
        changed["authorization"]["screen_metrics"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authorization"):
                load_calibration_execution(path, repo_root=REPO_ROOT, verify_artifacts=False)


if __name__ == "__main__":
    unittest.main()
