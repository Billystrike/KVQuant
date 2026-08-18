import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cases import token_ids_sha256
from utils.llama2_cage_v3_transfer_quality_manifest import (
    EXPECTED_ARTIFACT_MANIFEST_SHA256,
    Llama2CageV3TransferQualityManifestError,
    build_input_manifest,
    load_static_preflight_artifacts,
    validate_input_manifest,
)
from utils.llama2_cage_v3_transfer_quality_protocol import (
    EXPECTED_PROTOCOL_SHA256,
    load_transfer_quality_protocol,
)
from utils.qwen3_cage_v4_data import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_protocol_v1.json"
ARTIFACT_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_static_preflight_artifacts_v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "llama2_build_cage_v3_transfer_quality_input_manifest.py"


class Llama2CageV3TransferQualityManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha = load_transfer_quality_protocol(
            PROTOCOL_PATH, repo_root=REPO_ROOT
        )

    def test_static_preflight_artifact_manifest_is_exact_and_closed(self):
        self.assertEqual(file_sha256(ARTIFACT_PATH), EXPECTED_ARTIFACT_MANIFEST_SHA256)
        artifacts, preflight = load_static_preflight_artifacts(
            ARTIFACT_PATH, verify_server_files=False
        )
        self.assertIsNone(preflight)
        self.assertTrue(artifacts["decision"]["exact_input_manifest_build_authorized"])
        self.assertFalse(artifacts["decision"]["quality_metrics_authorized"])
        self.assertFalse(artifacts["decision"]["full_600_case_execution_authorized"])

    def _synthetic_manifest(self):
        token_ids = [index % 32000 for index in range(341468)]
        protocol = copy.deepcopy(self.protocol)
        protocol["input"]["expected_token_ids_sha256"] = token_ids_sha256(token_ids)
        snapshot = {
            "rows": 4358,
            "nonempty_rows": 2891,
            "joined_utf8_bytes": 1296370,
            "joined_text_sha256": "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83",
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "dataset_fingerprint": "synthetic",
            "datasets_version": "test",
        }
        manifest = build_input_manifest(
            protocol=protocol,
            protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            token_ids=token_ids,
            corpus_snapshot=snapshot,
            tokenizer_identity={
                "implementation": "test.Tokenizer",
                "use_fast": False,
                "files": {
                    "tokenizer_config.json": {"sha256": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a"},
                    "tokenizer.json": {"sha256": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e"},
                    "tokenizer.model": {"sha256": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"},
                    "special_tokens_map.json": {"sha256": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0"},
                },
            },
            source_state={"git_commit": "f" * 40, "dirty": False},
        )
        return protocol, token_ids, manifest

    def test_builds_exact_50_anchor_150_input_600_expansion_manifest(self):
        protocol, token_ids, manifest = self._synthetic_manifest()
        self.assertEqual(manifest["selection"]["anchor_count"], 50)
        self.assertEqual(manifest["selection"]["input_record_count"], 150)
        self.assertEqual(manifest["selection"]["expanded_method_case_count"], 600)
        self.assertEqual(len(manifest["inputs"]), 150)
        self.assertEqual(len({row["input_id"] for row in manifest["inputs"]}), 150)
        validate_input_manifest(manifest, protocol=protocol, token_ids=token_ids)

    def test_rejects_token_hash_window_or_execution_boundary_mutation(self):
        protocol, token_ids, manifest = self._synthetic_manifest()
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["inputs"][0]["prompt_ids"][0] += 1
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["inputs"][0]["identity"]["continuation_start"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["boundary"]["quality_acceptance_execution_authorized"] = True
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(Llama2CageV3TransferQualityManifestError):
                    validate_input_manifest(changed, protocol=protocol, token_ids=token_ids)

    def test_builder_loads_tokenizer_and_dataset_but_never_model_weights_or_quality(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("AutoTokenizer", source)
        self.assertIn("load_dataset", source)
        for forbidden in (
            "import torch",
            "AutoModel",
            "LlamaForCausalLM",
            "from_pretrained(\n        str(model),\n        torch_dtype",
            "mean_nll",
            "cross_entropy",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
