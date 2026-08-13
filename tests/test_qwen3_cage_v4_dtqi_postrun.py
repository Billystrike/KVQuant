import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.qwen3_cage_v4_data import canonical_sha256
from utils.qwen3_cage_v4_dtqi_acceptance import SCIENTIFIC_FIELDS
from utils.qwen3_cage_v4_dtqi_postrun import (
    build_postrun_audit,
    validate_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_gpu_acceptance_artifacts_v1.json"


class CageV4DTQIPostrunTest(unittest.TestCase):
    def test_checked_in_artifact_manifest_is_frozen(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_artifact_manifest(manifest)
        self.assertEqual(set(manifest["repeats"]), {"a", "b"})
        self.assertFalse(manifest["execution_boundary"]["gpu_full_screen_authorized"])

    def test_artifact_manifest_rejects_premature_full_screen(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(manifest)
        changed["execution_boundary"]["gpu_full_screen_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "execution boundary changed"):
            validate_artifact_manifest(changed)

    def test_build_audit_requires_bitwise_equal_repeat_payloads(self):
        payload = [
            {
                "case_id": "case-a",
                "method": {},
                "input": {},
                "memory": {},
                "scoring": {},
                "cache": {},
                "resume": {},
            }
        ]
        repeat_record = {
            "case_count": 3,
            "failure_count": 0,
            "scientific_payload_sha256": canonical_sha256(payload),
        }
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        comparison = {
            "status": "pass",
            "mismatch_case_ids": [],
            "case_count": 3,
            "fields": list(SCIENTIFIC_FIELDS),
            "required_consistency": "bitwise_equal_json_numeric_payload",
            "scientific_payload_sha256": canonical_sha256(payload),
            "comparator_sha256": manifest["comparison"]["comparator_sha256"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            comparison_path = root / "comparison.json"
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            manifest["comparison"]["path"] = str(comparison_path)
            import hashlib
            manifest["comparison"]["sha256"] = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
            manifest["comparison"]["size_bytes"] = comparison_path.stat().st_size
            manifest["comparison"]["scientific_payload_sha256"] = canonical_sha256(payload)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "utils.qwen3_cage_v4_dtqi_postrun.validate_artifact_manifest"
            ), patch(
                "utils.qwen3_cage_v4_dtqi_postrun.validate_repeat",
                side_effect=[(repeat_record, payload), (repeat_record, payload)],
            ):
                audit = build_postrun_audit(manifest_path)
            self.assertEqual(audit["status"], "pass")
            self.assertTrue(audit["repeat_payloads_bitwise_equal"])
            second = copy.deepcopy(payload)
            second[0]["scoring"] = {"mean_nll": 2.0}
            with patch(
                "utils.qwen3_cage_v4_dtqi_postrun.validate_artifact_manifest"
            ), patch(
                "utils.qwen3_cage_v4_dtqi_postrun.validate_repeat",
                side_effect=[(repeat_record, payload), (repeat_record, second)],
            ):
                with self.assertRaisesRegex(RuntimeError, "payloads differ"):
                    build_postrun_audit(manifest_path)


if __name__ == "__main__":
    unittest.main()
