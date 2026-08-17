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

    def test_manifest_mutation_is_rejected_before_artifact_reads(self) -> None:
        with mock.patch(
            "utils.llama2_cage_v3_gpu_acceptance_postrun.file_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaises(Llama2CageV3GPUExecutionError):
                build_postrun_audit(self.path)


if __name__ == "__main__":
    unittest.main()
