from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]

from utils.llama2_cage_v3_gpu_acceptance_postrun import EXPECTED_MANIFEST_SHA256, build_postrun_audit
from utils.llama2_cage_v3_gpu_execution import Llama2CageV3GPUExecutionError
from utils.qwen3_cage_v4_data import file_sha256


class Llama2CageV3GPUAcceptancePostrunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO_ROOT / "configs/llama2_7b_cage_v3_gpu_acceptance_artifacts_v1.json"

    def test_checked_in_artifact_manifest_is_frozen(self) -> None:
        self.assertEqual(file_sha256(self.path), EXPECTED_MANIFEST_SHA256)
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["claim_eligible"])
        self.assertFalse(manifest["preserved_boundary"]["formal_transfer_authorized"])
        self.assertFalse(manifest["preserved_boundary"]["quality_metric_computed"])

    def test_failed_wrapper_attempt_is_preserved_without_scientific_rerun(self) -> None:
        path = REPO_ROOT / "configs/llama2_7b_cage_v3_gpu_acceptance_postrun_failed_attempt_v1.json"
        self.assertEqual(file_sha256(path), "9a6c645ab2127e3b5202396fa419ca62e7979c0d021f6d7dd78ea8d48139820d")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(receipt["scientific_artifacts_changed"])
        self.assertFalse(receipt["gpu_acceptance_rerun_required"])
        self.assertTrue(receipt["postrun_read_only_retry_authorized"])

    def test_postrun_validator_does_not_import_torch_or_load_model(self) -> None:
        source = (REPO_ROOT / "scripts/llama2_validate_cage_v3_gpu_acceptance_postrun.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        self.assertNotIn("from_pretrained", source)

    def test_log_validation_respects_tee_scope(self) -> None:
        source = (REPO_ROOT / "utils/llama2_cage_v3_gpu_acceptance_postrun.py").read_text(encoding="utf-8")
        self.assertIn('"ACCEPTANCE_STATUS=0"', source)
        self.assertIn('"COMPARATOR_STATUS=0"', source)
        self.assertNotIn('"RUN_STATUS=0"', source)
        self.assertNotIn('"TEE_STATUS=0"', source)
        self.assertNotIn("_RESULT=PASS", source)

    def test_manifest_mutation_is_rejected_before_artifact_reads(self) -> None:
        with mock.patch(
            "utils.llama2_cage_v3_gpu_acceptance_postrun.file_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaises(Llama2CageV3GPUExecutionError):
                build_postrun_audit(self.path)


if __name__ == "__main__":
    unittest.main()
