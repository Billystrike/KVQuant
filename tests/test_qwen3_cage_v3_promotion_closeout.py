import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_closeout import (
    BASELINES,
    CageV3PromotionCloseoutError,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_analysis_artifacts_v1.json"


class CageV3PromotionCloseoutTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_freezes_negative_promotion_and_all_three_comparisons(self):
        validate_manifest(self.manifest, verify_artifacts=False)
        self.assertFalse(self.manifest["frozen_decision"]["promotion_pass"])
        self.assertEqual(
            self.manifest["frozen_decision"]["candidate_outcome"],
            "retain_original_cage_and_close_cage_v3_promotion",
        )
        self.assertEqual(tuple(self.manifest["frozen_comparisons"]), BASELINES)
        self.assertTrue(
            self.manifest["frozen_comparisons"]["cage-v1-kittypro-matched"]["baseline_pass"]
        )
        self.assertTrue(
            self.manifest["frozen_comparisons"]["kivi-kittypro-matched"]["baseline_pass"]
        )
        self.assertFalse(
            self.manifest["frozen_comparisons"]["kitty-pro-25pct"]["baseline_pass"]
        )

    def test_manifest_rejects_reopening_tuning_llama_or_test_access(self):
        mutations = []
        for key in (
            "additional_v3_tuning",
            "staged_llama2_v3_acceptance",
            "llama2_v3_full_experiments",
            "kitty_llama_port",
            "pg19_test_access",
        ):
            changed = copy.deepcopy(self.manifest)
            changed["closeout_boundary"][key] = True
            mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(CageV3PromotionCloseoutError):
                validate_manifest(changed, verify_artifacts=False)

    def test_manifest_rejects_hiding_unfavorable_4032_kitty_result(self):
        changed = copy.deepcopy(self.manifest)
        changed["frozen_comparisons"]["kitty-pro-25pct"][
            "per_length_relative_ppl_percent"
        ]["4032"] = -1.1904321873860653
        with self.assertRaisesRegex(CageV3PromotionCloseoutError, "frozen payload"):
            validate_manifest(changed, verify_artifacts=False)

    def test_closeout_runner_has_no_gpu_analysis_or_experiment_path(self):
        source = (REPO_ROOT / "scripts" / "qwen3_close_cage_v3_promotion.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch", source)
        self.assertNotIn("qwen3_analyze", source)
        self.assertNotIn("--stage", source)
        self.assertNotIn("input-manifest", source)


if __name__ == "__main__":
    unittest.main()
