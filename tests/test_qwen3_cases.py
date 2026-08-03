import unittest

from utils.qwen3_cases import (
    QWEN3_ANCHOR_COUNT,
    QWEN3_CONTINUATION_TOKENS,
    QWEN3_PROMPT_LENGTHS,
    build_anchor_records,
    continuation_anchor,
    minimum_anchor_gap,
    token_ids_sha256,
)


class Qwen3CasesTest(unittest.TestCase):
    def test_anchor_formula_uses_interior_51sts(self):
        token_count = 341468

        self.assertEqual(continuation_anchor(token_count, 0), 10647)
        self.assertEqual(continuation_anchor(token_count, 49), 334788)

    def test_builds_fifty_nonoverlapping_anchor_records(self):
        token_ids = list(range(350000))

        anchors = build_anchor_records(token_ids)

        self.assertEqual(len(anchors), QWEN3_ANCHOR_COUNT)
        self.assertGreaterEqual(
            minimum_anchor_gap(anchors),
            max(QWEN3_PROMPT_LENGTHS) + QWEN3_CONTINUATION_TOKENS,
        )
        self.assertEqual(
            [entry["prompt_length"] for entry in anchors[0]["prompts"]],
            list(QWEN3_PROMPT_LENGTHS),
        )
        self.assertEqual(
            len(token_ids[
                anchors[-1]["continuation_start"] :
                anchors[-1]["continuation_start"] + QWEN3_CONTINUATION_TOKENS
            ]),
            QWEN3_CONTINUATION_TOKENS,
        )

    def test_token_hash_is_canonical_and_order_sensitive(self):
        self.assertEqual(
            token_ids_sha256([1, 2, 3]),
            "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4",
        )
        self.assertNotEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([3, 2, 1]))

    def test_rejects_short_stream_and_invalid_tokens(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            continuation_anchor(100, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            token_ids_sha256([1, -1])


if __name__ == "__main__":
    unittest.main()
