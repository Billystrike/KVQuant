import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_formal import (
    Qwen3FormalError,
    build_input_manifest,
    load_formal_protocol,
    validate_input_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_formal_quality_protocol_v1.json"


class Qwen3FormalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha256 = load_formal_protocol(PROTOCOL_PATH)
        cls.token_ids = list(range(cls.protocol["input"]["token_count"]))
        from utils.qwen3_cases import token_ids_sha256

        cls.synthetic_protocol = copy.deepcopy(cls.protocol)
        cls.synthetic_protocol["input"]["token_ids_sha256"] = token_ids_sha256(
            cls.token_ids
        )

    def _manifest(self):
        return build_input_manifest(
            protocol=self.synthetic_protocol,
            protocol_sha256=self.protocol_sha256,
            token_ids=self.token_ids,
            corpus_snapshot_sha256=self.synthetic_protocol["input"][
                "corpus_snapshot_sha256"
            ],
            tokenizer_identity={"class": "synthetic"},
            source_state={"git_commit": "a" * 40, "dirty": False},
        )

    def test_builds_ordered_150_case_manifest(self):
        manifest = self._manifest()
        self.assertEqual(len(manifest["cases"]), 150)
        self.assertEqual(manifest["cases"][0]["input_case_id"], "qwen3-a00-l1024")
        self.assertEqual(manifest["cases"][-1]["input_case_id"], "qwen3-a49-l4032")
        self.assertEqual(len(manifest["cases"][0]["continuation_ids"]), 64)
        self.assertEqual(len(manifest["cases"][-1]["prompt_ids"]), 4032)

    def test_manifest_round_trip_validation(self):
        manifest = json.loads(json.dumps(self._manifest()))
        validate_input_manifest(
            manifest,
            protocol=self.synthetic_protocol,
            protocol_sha256=self.protocol_sha256,
        )

    def test_rejects_token_or_identity_mutation(self):
        manifest = self._manifest()
        manifest["cases"][0]["prompt_ids"][0] += 1
        with self.assertRaisesRegex(Qwen3FormalError, "token hash"):
            validate_input_manifest(
                manifest,
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
            )

        manifest = self._manifest()
        manifest["cases"][1]["input_case_id"] = manifest["cases"][0]["input_case_id"]
        with self.assertRaisesRegex(Qwen3FormalError, "ID/order"):
            validate_input_manifest(
                manifest,
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
            )

    def test_rejects_dirty_source_state(self):
        with self.assertRaisesRegex(Qwen3FormalError, "clean"):
            build_input_manifest(
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
                token_ids=self.token_ids,
                corpus_snapshot_sha256=self.synthetic_protocol["input"][
                    "corpus_snapshot_sha256"
                ],
                tokenizer_identity={"class": "synthetic"},
                source_state={"git_commit": "a" * 40, "dirty": True},
            )


if __name__ == "__main__":
    unittest.main()
