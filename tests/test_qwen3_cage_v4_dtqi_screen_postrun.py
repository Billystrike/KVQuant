import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_dtqi_screen_postrun import validate_artifact_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_artifacts_v1.json"


class CageV4DTQIScreenPostrunTest(unittest.TestCase):
    def test_checked_in_artifact_manifest_freezes_completed_screen(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_artifact_manifest(value)
        self.assertEqual(value["expected"]["case_count"], 120)
        self.assertFalse(value["execution_boundary"]["holdout_accessed"])
        self.assertFalse(value["execution_boundary"]["interpretation_performed"])

    def test_artifact_manifest_rejects_case_or_boundary_mutation(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        changed = copy.deepcopy(value)
        changed["expected"]["case_count"] = 119
        with self.assertRaisesRegex(RuntimeError, "expected counts changed"):
            validate_artifact_manifest(changed)
        changed = copy.deepcopy(value)
        changed["execution_boundary"]["holdout_accessed"] = True
        with self.assertRaisesRegex(RuntimeError, "boundary changed"):
            validate_artifact_manifest(changed)

    def test_validator_does_not_interpret_results(self):
        source = (REPO_ROOT / "scripts" / "qwen3_validate_cage_v4_dtqi_screen_postrun.py").read_text(encoding="utf-8")
        self.assertIn('"interpretation_performed": False', source)
        self.assertNotIn("bootstrap", source.lower())
        self.assertNotIn("relative_ppl", source.lower())


if __name__ == "__main__":
    unittest.main()
