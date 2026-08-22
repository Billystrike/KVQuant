import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_full_postrun import (
    EXPECTED_SCIENTIFIC_SHA256,
    shell_case_manifest_sha256,
    validate_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_full_artifacts_v1.json"


class Llama2CageV3TransferQualityFullPostrunTest(unittest.TestCase):
    def test_checked_in_manifest_freezes_complete_full_run_without_interpretation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_artifact_manifest(manifest, repo_root=REPO_ROOT)
        self.assertEqual(manifest["full_run"]["case_count"], 600)
        self.assertEqual(manifest["full_run"]["scientific_payload_sha256"], EXPECTED_SCIENTIFIC_SHA256)
        self.assertEqual(manifest["full_run"]["failure_count"], 0)
        self.assertFalse(manifest["authorization_before_postrun"]["quality_interpretation"])

    def test_manifest_rejects_payload_failure_or_authorization_mutation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["full_run"]["scientific_payload_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["full_run"]["failure_count"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["authorization_before_postrun"]["quality_interpretation"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                validate_artifact_manifest(mutation, repo_root=REPO_ROOT)

    def test_shell_case_manifest_reproduces_absolute_path_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cases").mkdir()
            (root / "cases" / "b.json").write_text("b\n", encoding="utf-8")
            (root / "cases" / "a.json").write_text("a\n", encoding="utf-8")
            lines = "".join(
                f"{hashlib.sha256((root / 'cases' / name).read_bytes()).hexdigest()}  {root / 'cases' / name}\n"
                for name in ("a.json", "b.json")
            )
            self.assertEqual(shell_case_manifest_sha256(root), hashlib.sha256(lines.encode()).hexdigest())

    def test_postrun_is_read_only_and_does_not_aggregate_quality(self):
        validator = (REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_quality_full_postrun.py").read_text(encoding="utf-8")
        utility = (REPO_ROOT / "utils" / "llama2_cage_v3_transfer_quality_full_postrun.py").read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained", validator)
        self.assertNotIn("from_pretrained", utility)
        self.assertNotIn("import torch", validator)
        self.assertNotIn("mean_nll_delta", utility)
        self.assertNotIn("bootstrap", utility)
        self.assertIn('"interpretation_performed": False', utility)
        self.assertIn('"quality_interpretation_authorized_by_this_audit": False', utility)


if __name__ == "__main__":
    unittest.main()
