import json
import unittest
from pathlib import Path

from utils.qwen3_memory import (
    closest_memory_candidate,
    estimate_qwen3_cage_bytes,
    estimate_qwen3_fp16_bytes,
    estimate_qwen3_kivi_bytes,
    estimate_qwen3_kitty_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


class Qwen3MemoryTest(unittest.TestCase):
    def test_fp16_counts_key_and_value_model_wide(self):
        report = estimate_qwen3_fp16_bytes(seq_len=320)

        self.assertEqual(report["per_layer"]["key_fp16_bytes"], 655360)
        self.assertEqual(report["model_total_bytes"], 47185920)

    def test_kitty_page_layout_and_320_token_probe_match_frozen_audit(self):
        kitty = estimate_qwen3_kitty_bytes(seq_len=320, boosted_channels=16)
        kitty_pro = estimate_qwen3_kitty_bytes(seq_len=320, boosted_channels=32)

        self.assertEqual(kitty["page_layout"]["combined_page_bytes"], 78848)
        self.assertEqual(kitty_pro["page_layout"]["combined_page_bytes"], 82944)
        self.assertEqual(kitty["page_layout"]["steady_bytes_per_token_model_wide"], 22176.0)
        self.assertEqual(kitty_pro["page_layout"]["steady_bytes_per_token_model_wide"], 23328.0)
        self.assertEqual(kitty["model_total_bytes"], 23225184)
        self.assertEqual(kitty_pro["model_total_bytes"], 23520096)
        self.assertEqual(
            kitty["token_state"],
            {
                "sink_tokens": 32,
                "key_page_count": 2,
                "key_buffer_tokens": 32,
                "value_page_count": 1,
                "value_buffer_tokens": 32,
                "value_local_tokens": 128,
            },
        )

    def test_kitty_targets_match_every_frozen_protocol_length(self):
        expected = {
            (1024, 16): 46857888,
            (1024, 32): 47890080,
            (2048, 16): 69570720,
            (2048, 32): 71782560,
            (4032, 16): 105559200,
            (4032, 32): 110130336,
        }
        for (seq_len, boosted), target in expected.items():
            with self.subTest(seq_len=seq_len, boosted=boosted):
                report = estimate_qwen3_kitty_bytes(
                    seq_len=seq_len, boosted_channels=boosted
                )
                self.assertEqual(report["model_total_bytes"], target)

    def test_cage_and_kivi_selected_bytes_match_frozen_protocol(self):
        cage_expected = {
            (1024, 224): 48253824,
            (2048, 224): 68684544,
            (2048, 288): 72518400,
            (4032, 32): 106206336,
            (4032, 96): 110040192,
        }
        for (seq_len, residual), target in cage_expected.items():
            with self.subTest(method="cage", seq_len=seq_len, residual=residual):
                report = estimate_qwen3_cage_bytes(
                    seq_len=seq_len, residual_length=residual
                )
                self.assertEqual(report["model_total_bytes"], target)

        kivi_expected = {
            (1024, 64, 320): 47480832,
            (2048, 32, 224): 71958528,
            (4032, 64, 128): 104841216,
            (4032, 128, 256): 111992832,
        }
        for (seq_len, group, residual), target in kivi_expected.items():
            with self.subTest(method="kivi", seq_len=seq_len, residual=residual):
                report = estimate_qwen3_kivi_bytes(
                    seq_len=seq_len,
                    group_size=group,
                    residual_length=residual,
                )
                self.assertEqual(report["model_total_bytes"], target)

    def test_byte_only_selection_reproduces_frozen_matrix(self):
        protocol = json.loads(
            (ROOT / "configs" / "qwen3_8b_matched_memory_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        tolerance = protocol["matched_memory_matrix"]["relative_budget_tolerance"]
        for frozen in protocol["matched_memory_matrix"]["selections"]:
            seq_len = frozen["prompt_length"]
            target = frozen["target_bytes"]
            cage_candidates = [
                estimate_qwen3_cage_bytes(seq_len=seq_len, residual_length=residual)
                for residual in range(32, 513, 32)
            ]
            kivi_candidates = [
                estimate_qwen3_kivi_bytes(
                    seq_len=seq_len,
                    group_size=group,
                    residual_length=residual,
                )
                for group in (32, 64, 128)
                for residual in range(group, 513, group)
            ]
            cage, cage_delta = closest_memory_candidate(cage_candidates, target)
            kivi, kivi_delta = closest_memory_candidate(kivi_candidates, target)

            with self.subTest(seq_len=seq_len, target=frozen["target_method"]):
                self.assertEqual(cage["method"], frozen["cage_method"])
                self.assertEqual(cage["model_total_bytes"], frozen["cage_bytes"])
                self.assertAlmostEqual(cage_delta, frozen["cage_relative_delta"])
                if frozen["kivi_method"] is None:
                    self.assertGreater(abs(kivi_delta), tolerance)
                else:
                    self.assertEqual(kivi["method"], frozen["kivi_method"])
                    self.assertEqual(kivi["model_total_bytes"], frozen["kivi_bytes"])
                    self.assertAlmostEqual(kivi_delta, frozen["kivi_relative_delta"])

    def test_rejects_invalid_layouts(self):
        with self.assertRaisesRegex(ValueError, "multiple of group_size"):
            estimate_qwen3_kivi_bytes(
                seq_len=320, group_size=64, residual_length=96
            )
        with self.assertRaisesRegex(ValueError, "sum to head_dim"):
            estimate_qwen3_cage_bytes(
                seq_len=320,
                residual_length=128,
                key_bucket_sizes=(42, 43, 42),
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            estimate_qwen3_kitty_bytes(seq_len=320, boosted_channels=129)


if __name__ == "__main__":
    unittest.main()
