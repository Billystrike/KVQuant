import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_dtqi_closeout import validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_analysis_artifacts_v1.json"


class CageV4DTQICloseoutTest(unittest.TestCase):
    def test_checked_in_manifest_freezes_negative_decision(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_manifest(value, verify_artifacts=False)
        self.assertEqual(value["frozen_decision"]["candidate_outcome"], "close_cage_v4_as_negative")
        self.assertFalse(value["closeout_boundary"]["additional_tuning_on_same_screen"])
        self.assertFalse(value["closeout_boundary"]["holdout_metrics_authorized"])

    def test_manifest_rejects_reinterpretation_or_holdout(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        changed = copy.deepcopy(value)
        changed["frozen_decision"]["candidate_outcome"] = "advance_to_holdout_preflight"
        with self.assertRaisesRegex(RuntimeError, "frozen decision changed"):
            validate_manifest(changed, verify_artifacts=False)
        changed = copy.deepcopy(value)
        changed["closeout_boundary"]["holdout_metrics_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "boundary changed"):
            validate_manifest(changed, verify_artifacts=False)

    def test_closeout_runner_has_no_gpu_or_holdout_path(self):
        source = (REPO_ROOT / "scripts" / "qwen3_close_cage_v4_dtqi.py").read_text(encoding="utf-8")
        self.assertNotIn("torch", source)
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("holdout_execution", source)


if __name__ == "__main__":
    unittest.main()
