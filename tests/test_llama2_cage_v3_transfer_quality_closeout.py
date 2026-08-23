import copy
import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_closeout import validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_analysis_artifacts_v1.json"


class Llama2CageV3TransferQualityCloseoutTest(unittest.TestCase):
    def test_checked_in_manifest_freezes_nonpromotion_and_secondary_role(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_manifest(value, repo_root=REPO_ROOT, verify_artifacts=False)
        self.assertFalse(value["frozen_decision"]["promotion_pass"])
        self.assertEqual(value["frozen_decision"]["llama2_main_method"], "cage_v1")
        self.assertEqual(value["frozen_decision"]["cage_v3_role"], "secondary_length_dependent_cross_architecture_refinement")

    def test_manifest_rejects_gate_relaxation_tuning_or_suppressed_unfavorable_results(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(value)
        changed["closeout_boundary"]["relax_frozen_promotion_gate"] = True
        mutations.append(changed)
        changed = copy.deepcopy(value)
        changed["closeout_boundary"]["additional_tuning_on_same_anchors"] = True
        mutations.append(changed)
        changed = copy.deepcopy(value)
        changed["closeout_boundary"]["report_all_lengths_and_unfavorable_results"] = False
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                validate_manifest(mutation, repo_root=REPO_ROOT, verify_artifacts=False)

    def test_closeout_runner_is_read_only_and_has_no_gpu_or_raw_case_analysis(self):
        runner = (REPO_ROOT / "scripts" / "llama2_close_cage_v3_transfer_quality.py").read_text(encoding="utf-8")
        utility = (REPO_ROOT / "utils" / "llama2_cage_v3_transfer_quality_closeout.py").read_text(encoding="utf-8")
        self.assertNotIn("torch", runner)
        self.assertNotIn("from_pretrained", utility)
        self.assertNotIn("import random", utility)
        self.assertNotIn("def _percentile", utility)
        self.assertNotIn("def bootstrap_draws", utility)
        self.assertNotIn("token_nlls", utility)


if __name__ == "__main__":
    unittest.main()
