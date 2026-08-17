from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_cpu_acceptance_postrun import (
    Llama2CageV3CPUPostrunError,
    validate_artifact_manifest,
)
from utils.qwen3_cage_v4_data import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_cpu_acceptance_artifacts_v1.json"


class Llama2CageV3CPUAcceptancePostrunTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_checked_in_artifact_manifest_preserves_scientific_pass_and_wrapper_failure(self):
        validate_artifact_manifest(self.manifest, repo_root=REPO_ROOT)
        self.assertEqual(
            file_sha256(MANIFEST_PATH),
            "1dde892fa6f057a827773c263e1cb1e82eb94355a4a338229399e7b444a6152e",
        )
        self.assertTrue(self.manifest["wrapper_outcome"]["scientific_acceptance_completed_before_failure"])
        self.assertFalse(self.manifest["wrapper_outcome"]["package_check_pass"])
        repair = self.manifest["postrun_administrative_repair"]
        self.assertEqual(
            file_sha256(REPO_ROOT / repair["failed_attempt_receipt_path"]),
            repair["failed_attempt_receipt_sha256"],
        )

    def test_manifest_rejects_hiding_package_failure_or_authorizing_gpu_execution(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["wrapper_outcome"]["package_check_pass"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUPostrunError, "package-check"):
            validate_artifact_manifest(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.manifest)
        mutated["postrun_boundary"]["gpu_acceptance_execution_authorized"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUPostrunError, "gpu_acceptance_execution_authorized"):
            validate_artifact_manifest(mutated, repo_root=REPO_ROOT)

    def test_manifest_rejects_scientific_rerun_or_environment_mutation(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["wrapper_outcome"]["rerun_scientific_acceptance_required"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUPostrunError, "rerun"):
            validate_artifact_manifest(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.manifest)
        mutated["wrapper_outcome"]["environment_mutation_authorized"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUPostrunError, "environment mutation"):
            validate_artifact_manifest(mutated, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
