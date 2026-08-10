import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_holdout_postrun import (
    CageV3HoldoutPostrunError,
    validate_artifact_manifest,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_full_artifacts_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_execution_v1.json"
DECISION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_decision_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_acceptance_gate_v1.json"
EXPECTED_ARTIFACT_SHA256 = "25ea16b8d43d3e88b26d2585c7553eab1b3e11f7f76a6efe7270631247523a13"


class CageV3HoldoutPostrunTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.kwargs = {
            "execution_sha256": file_sha256(EXECUTION_PATH),
            "protocol_sha256": self.artifacts["protocol_sha256"],
            "input_manifest_sha256": self.artifacts["manifest_sha256"],
            "quota_plan_sha256": self.artifacts["quota_plan_sha256"],
            "screen_decision_sha256": file_sha256(DECISION_PATH),
            "gate_sha256": file_sha256(GATE_PATH),
        }

    def test_artifact_declaration_freezes_exact_90_case_uninterpreted_run(self):
        self.assertEqual(file_sha256(ARTIFACT_PATH), EXPECTED_ARTIFACT_SHA256)
        validate_artifact_manifest(self.artifacts, **self.kwargs)
        self.assertFalse(self.artifacts["interpretation_performed"])
        self.assertEqual(self.artifacts["joint_expected"]["case_count"], 90)
        self.assertEqual(self.artifacts["joint_expected"]["layer_record_count"], 3240)
        self.assertTrue(self.artifacts["joint_expected"]["holdout_metrics_consumed"])
        self.assertFalse(self.artifacts["joint_expected"]["reserved_unseen_metrics_consumed"])
        self.assertEqual(self.artifacts["partitions"]["cage_qwen3"]["case_count"], 60)
        self.assertEqual(self.artifacts["partitions"]["kitty_qwen3"]["case_count"], 30)

    def test_artifact_declaration_rejects_interpretation_or_reserved_access(self):
        changed = copy.deepcopy(self.artifacts)
        changed["interpretation_performed"] = True
        with self.assertRaisesRegex(CageV3HoldoutPostrunError, "interpreted"):
            validate_artifact_manifest(changed, **self.kwargs)
        changed = copy.deepcopy(self.artifacts)
        changed["joint_expected"]["reserved_unseen_metrics_consumed"] = True
        with self.assertRaisesRegex(CageV3HoldoutPostrunError, "expectations"):
            validate_artifact_manifest(changed, **self.kwargs)

    def test_acceptance_scientific_sources_remain_bitwise_frozen(self):
        expected = {
            "scripts/qwen3_run_cage_v3_holdout.py": "bcb283a064d719fa816de2de5896238e16d44b3adb881a6eaa32164d2c429035",
            "scripts/qwen3_compare_cage_v3_holdout_acceptance.py": "390e9644487fb7e0199399fd7566c6df76c712a60b2de36dd49f76c292512f60",
            "utils/qwen3_cage_v3_holdout.py": "5fc61df7095a5dc2442d22e50b0992596e343209c0383e52b716c236afa0aed3",
        }
        for relative, sha256 in expected.items():
            self.assertEqual(file_sha256(REPO_ROOT / relative), sha256)


if __name__ == "__main__":
    unittest.main()
