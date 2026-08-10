import copy
import math
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_metric_protocol import (
    COMPRESSED_METHODS,
    load_metric_protocol,
    practical_effect_label,
    relative_ppl_percent,
    validate_metric_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_validity_protocol_v1.json"


class CageV4MetricProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_metric_protocol(PROTOCOL_PATH)

    def test_protocol_freezes_practical_effect_and_blocks_gpu_holdout_and_v4(self):
        self.assertEqual(len(self.protocol_sha256), 64)
        self.assertEqual(
            [method["method_id"] for method in self.protocol["method_grid"]],
            ["fp16", *COMPRESSED_METHODS],
        )
        self.assertEqual(self.protocol["execution_boundary"]["full_quality_cases"], 720)
        self.assertEqual(self.protocol["execution_boundary"]["compressed_perturbation_cases"], 600)
        self.assertFalse(self.protocol["execution_boundary"]["holdout_method_metrics_authorized"])
        self.assertFalse(self.protocol["execution_boundary"]["cage_v4_candidate_execution_authorized"])
        self.assertFalse(self.protocol["execution_boundary"]["full_gpu_execution_authorized"])

    def test_relative_ppl_and_practical_labels_match_frozen_interpretation(self):
        self.assertAlmostEqual(relative_ppl_percent(math.log(0.99)), -1.0)
        self.assertAlmostEqual(relative_ppl_percent(math.log(1.005)), 0.5)
        self.assertEqual(practical_effect_label(math.log(0.997)), "practically_equivalent")
        self.assertEqual(practical_effect_label(math.log(0.993)), "small_favorable_not_material")
        self.assertEqual(practical_effect_label(math.log(0.99)), "material_favorable")
        self.assertEqual(practical_effect_label(math.log(1.01)), "material_unfavorable")
        with self.assertRaises(ValueError):
            relative_ppl_percent(float("nan"))

    def test_protocol_rejects_posthoc_threshold_or_execution_mutation(self):
        changed = copy.deepcopy(self.protocol)
        changed["practical_effect_policy"]["material_absolute_relative_ppl_percent_at_least"] = 0.1
        with self.assertRaisesRegex(ValueError, "practical-effect"):
            validate_metric_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed["execution_boundary"]["holdout_method_metrics_authorized"] = True
        with self.assertRaisesRegex(ValueError, "execution boundary"):
            validate_metric_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed["local_proxy_validity_gate"]["minimum_spearman_each_length"] = 0.0
        with self.assertRaisesRegex(ValueError, "Spearman"):
            validate_metric_protocol(changed)

    def test_no_compressed_point_exceeds_kitty_pro_by_more_than_three_percent(self):
        methods = {method["method_id"]: method for method in self.protocol["method_grid"]}
        kitty = {row["prompt_length"]: row["packed_bytes"] for row in methods["kitty-pro-25pct"]["points"]}
        for method_id in COMPRESSED_METHODS:
            if method_id == "kitty-pro-25pct":
                continue
            for point in methods[method_id]["points"]:
                relative = point["packed_bytes"] / kitty[point["prompt_length"]] - 1.0
                self.assertLessEqual(relative, 0.03)


if __name__ == "__main__":
    unittest.main()
