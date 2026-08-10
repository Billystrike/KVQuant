import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_pg19 import (
    EXPECTED_SOURCE,
    audit_documents,
    canonical_sha256,
    load_source_candidate,
    validate_source_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_pg19_source_candidate_v1.json"


class _SyntheticTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs != {
            "add_special_tokens": False,
            "return_attention_mask": False,
            "return_token_type_ids": False,
            "truncation": False,
            "verbose": False,
        }:
            raise AssertionError("tokenizer controls changed")
        return {"input_ids": [ord(character) for character in text]}


class CageV4Pg19Test(unittest.TestCase):
    def test_checked_in_source_candidate_freezes_only_provenance(self):
        candidate, candidate_sha256 = load_source_candidate(SOURCE_PATH)
        self.assertEqual(len(candidate_sha256), 64)
        self.assertFalse(candidate["claim_eligible"])
        self.assertEqual(candidate["mirror_snapshot"]["revision"], EXPECTED_SOURCE["revision"])
        self.assertEqual(candidate["official_source"]["license_spdx"], "Apache-2.0")
        self.assertEqual(
            candidate["pre_download_audit"]["log_sha256"],
            "c7d7f1b9a6ac30b2b024e3f3287455e6c5759460f4009e4f6a8a0613f5102751",
        )
        self.assertEqual(
            candidate["mirror_snapshot"]["license_metadata_status"],
            "absent_in_hugging_face_repo_tags",
        )
        self.assertTrue(all(value is False for value in candidate["freeze_boundary"].values()))

    def test_source_candidate_rejects_revision_license_or_boundary_mutation(self):
        candidate = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        for path, value in (
            (("mirror_snapshot", "revision"), "0" * 40),
            (("official_source", "license_spdx"), "unknown"),
            (("pre_download_audit", "status"), "fail"),
            (("freeze_boundary", "documents_selected"), True),
        ):
            changed = copy.deepcopy(candidate)
            changed[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                validate_source_candidate(changed)

    def test_document_audit_is_deterministic_and_does_not_select_documents(self):
        rows = [
            {"text": f"book-{index}-" + ("x" * (4100 + index)), "title": f"Title {index}"}
            for index in range(50)
        ]
        first, first_summary = audit_documents(rows, tokenizer=_SyntheticTokenizer())
        second, second_summary = audit_documents(rows, tokenizer=_SyntheticTokenizer())
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first), 50)
        self.assertEqual(first_summary["eligible_document_counts"]["4096"], 50)
        self.assertEqual(first_summary["duplicate_text_sha256_groups"], [])
        self.assertEqual(first_summary["duplicate_token_ids_sha256_groups"], [])
        self.assertEqual(len(first[0]["document_id"]), len("pg19-validation-00-") + 16)

    def test_document_audit_rejects_wrong_count_empty_text_or_duplicate_detection(self):
        tokenizer = _SyntheticTokenizer()
        with self.assertRaisesRegex(ValueError, "50"):
            audit_documents([{"text": "x"}], tokenizer=tokenizer)
        rows = [{"text": f"book-{index}"} for index in range(50)]
        rows[7]["text"] = " "
        with self.assertRaisesRegex(ValueError, "row 7"):
            audit_documents(rows, tokenizer=tokenizer)
        rows = [{"text": f"book-{index}"} for index in range(50)]
        rows[49]["text"] = rows[0]["text"]
        _, summary = audit_documents(rows, tokenizer=tokenizer)
        self.assertEqual(summary["duplicate_text_sha256_groups"], [[0, 49]])
        self.assertEqual(summary["duplicate_token_ids_sha256_groups"], [[0, 49]])

    def test_canonical_hash_is_order_sensitive_and_json_stable(self):
        self.assertEqual(canonical_sha256({"b": 2, "a": 1}), canonical_sha256({"a": 1, "b": 2}))
        self.assertNotEqual(canonical_sha256([1, 2, 3]), canonical_sha256([3, 2, 1]))


if __name__ == "__main__":
    unittest.main()
