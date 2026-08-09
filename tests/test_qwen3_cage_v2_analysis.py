import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v2_analysis import (
    EXPECTED_RECEIPT_SHA256,
    build_round1_analysis,
    paired_statistics,
    validate_results_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v2_round1_results_receipt_v1.json"


class CageV2Round1AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_checked_in_receipt_hash_and_analysis_boundary_are_frozen(self):
        validate_results_receipt(self.receipt, receipt_path=RECEIPT_PATH)
        self.assertEqual(EXPECTED_RECEIPT_SHA256, self.receipt_sha())
        self.assertEqual(len(self.receipt["analysis_freeze"]["family_tracks"]), 6)
        self.assertEqual(self.receipt["analysis_freeze"]["maximum_distinct_families_advanced"], 3)

    def receipt_sha(self):
        import hashlib

        return hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()

    def test_paired_statistics_are_deterministic_and_directional(self):
        candidate = {5: 0.8, 6: 0.7, 7: 0.9, 8: 0.6, 9: 1.0}
        baseline = {anchor: 1.0 for anchor in candidate}
        first = paired_statistics(candidate, baseline, bootstrap_resamples=100, bootstrap_seed=17)
        second = paired_statistics(candidate, baseline, bootstrap_resamples=100, bootstrap_seed=17)
        self.assertEqual(first, second)
        self.assertLess(first["mean_delta"], 0)
        self.assertEqual(first["favor_count"], 4)
        self.assertEqual(first["tie_count"], 1)

    def test_family_advancement_uses_one_track_per_structure(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["analysis_freeze"]["bootstrap_resamples"] = 20
        records = []
        freeze = receipt["analysis_freeze"]
        method_values = {}
        for target, mapping in freeze["cage_v1_mapping"].items():
            for method_id in mapping.values():
                method_values[method_id] = 1.0
        for target, mapping in freeze["kitty_mapping"].items():
            for method_id in mapping.values():
                method_values[method_id] = 0.8
        for track in freeze["family_tracks"]:
            value = 0.7 if track["family_id"] == "sr2-no-sink" else 0.9
            if track["family_id"] == "sr1-sr2-sink32":
                value = 0.6
            for length in freeze["prompt_lengths"]:
                method_values[f"{track['method_prefix']}{length}"] = value
        for method_id, value in method_values.items():
            length = int(method_id.rsplit("l", 1)[1])
            name = "cage_v1" if method_id.startswith("cage-v1-") else (
                "kitty" if method_id.startswith("kitty-") else "cage_v2"
            )
            if method_id.startswith("kitty-pro-"):
                packed_bytes = 1100
            elif method_id.startswith("kitty-"):
                packed_bytes = 1000
            elif method_id.startswith("cage-v1-"):
                if length == 1024:
                    packed_bytes = 1050
                elif "r224-" in method_id or "r32-" in method_id:
                    packed_bytes = 990
                else:
                    packed_bytes = 1090
            else:
                packed_bytes = 900
            for anchor in freeze["anchor_indices"]:
                records.append(
                    {
                        "method": {"id": method_id, "name": name, "prompt_length": length},
                        "input": {"anchor_index": anchor},
                        "memory": {"model_total_bytes": packed_bytes},
                        "aggregates": {"joint_post_o_proj_mse": {"mean": value}},
                    }
                )
        analysis = build_round1_analysis(
            receipt=receipt,
            records=records,
            receipt_sha256="a" * 64,
        )
        self.assertEqual(analysis["advanced_family_count"], 2)
        self.assertEqual(
            {row["family_id"] for row in analysis["advanced_families"]},
            {"sr2-no-sink", "sr1-sr2-sink32"},
        )
        self.assertEqual(len({row["family_id"] for row in analysis["advanced_families"]}), 2)


if __name__ == "__main__":
    unittest.main()
