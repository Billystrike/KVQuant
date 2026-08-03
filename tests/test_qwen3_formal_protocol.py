import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_formal_quality_protocol_v1.json"


class Qwen3FormalProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_base_and_mechanism_case_counts_are_exact(self):
        anchors = self.protocol["input"]["anchor_count"]
        base_points = sum(
            len(record["prompt_lengths"])
            for record in self.protocol["base_method_length_points"]
        )
        additional_mechanism_points = (
            len(self.protocol["mechanism_points"])
            * len(self.protocol["mechanism_variants"])
        )
        counts = self.protocol["case_counts"]

        self.assertEqual(base_points, 18)
        self.assertEqual(counts["base_method_length_points_per_anchor"], base_points)
        self.assertEqual(counts["base_cases"], anchors * base_points)
        self.assertEqual(
            counts["additional_mechanism_points_per_anchor"],
            additional_mechanism_points,
        )
        self.assertEqual(
            counts["additional_mechanism_cases"],
            anchors * additional_mechanism_points,
        )
        self.assertEqual(
            counts["total_unique_method_length_cases"],
            counts["base_cases"] + counts["additional_mechanism_cases"],
        )
        self.assertEqual(
            counts["nll_targets"],
            counts["total_unique_method_length_cases"]
            * self.protocol["scoring"]["target_count_per_case"],
        )

    def test_primary_comparisons_reference_executable_base_points(self):
        executable = {
            (record["method_id"], prompt_length)
            for record in self.protocol["base_method_length_points"]
            for prompt_length in record["prompt_lengths"]
        }
        comparisons = self.protocol["primary_matched_comparisons"]

        self.assertEqual(len(comparisons), 10)
        for comparison in comparisons:
            key = (comparison["candidate"], comparison["prompt_length"])
            baseline = (comparison["baseline"], comparison["prompt_length"])
            self.assertIn(key, executable)
            self.assertIn(baseline, executable)

    def test_mechanism_controls_change_only_assignment_policy(self):
        invariant = self.protocol["mechanism_memory_invariant"]
        self.assertTrue(invariant["exact_same_packed_bytes_as_full_cage"])
        self.assertEqual(
            invariant["only_changed_field"],
            "data-dependent versus fixed channel ordering",
        )
        variants = {
            record["suffix"]: (
                record["key_importance"],
                record["value_importance"],
            )
            for record in self.protocol["mechanism_variants"]
        }
        self.assertEqual(
            variants,
            {
                "fixed-random": ("fixed_random", "fixed_random"),
                "fixed-uniform": ("fixed_uniform", "fixed_uniform"),
                "key-adaptive-only": ("q2_var", "fixed_uniform"),
                "value-adaptive-only": ("fixed_uniform", "wo_var"),
            },
        )

    def test_case_windows_fit_native_context_and_do_not_overlap(self):
        maximum_window = (
            max(self.protocol["input"]["prompt_lengths"])
            + self.protocol["input"]["continuation_tokens"]
        )
        self.assertLessEqual(maximum_window, self.protocol["model"]["native_context"])
        self.assertGreaterEqual(self.protocol["input"]["minimum_anchor_gap"], maximum_window)


if __name__ == "__main__":
    unittest.main()
