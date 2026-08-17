from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from utils.llama2_cage_v3_gpu_execution import (
    Llama2CageV3GPUExecutionError,
    validate_gate_receipt,
    validate_static_preflight_receipt_file,
)
from utils.qwen3_cage_v4_data import file_sha256


class Llama2CageV3GPUExecutionTest(unittest.TestCase):
    def test_execution_source_manifest_is_frozen_before_gpu_run(self) -> None:
        path = REPO_ROOT / "configs/llama2_7b_cage_v3_gpu_execution_sources_v1.json"
        self.assertEqual(file_sha256(path), "5a67eb0aeef944cc2ccda7516e6b88d9ddd803cd0bdabfa1385799aa6d604f4d")

    def test_checked_in_static_preflight_receipt_is_exact_and_passed(self) -> None:
        path = REPO_ROOT / "configs/llama2_7b_cage_v3_gpu_static_preflight_receipt_v1.json"
        receipt = validate_static_preflight_receipt_file(path)
        self.assertEqual(file_sha256(path), "63f1473f02a7628ebe677e6e65b7001870514c6f81a630ffb1e3131b155b2d25")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertFalse(any(receipt["execution_boundary"].values()))

    def test_checked_in_execution_gate_receipt_matches_server_artifact(self) -> None:
        path = REPO_ROOT / "configs/llama2_7b_cage_v3_gpu_execution_gate_receipt_v1.json"
        self.assertEqual(file_sha256(path), "24054c0018578e6b8eb07e69bd931f89b49871476b4ef397243ceff607d2daff")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "pass")
        self.assertTrue(receipt["decision"]["gpu_acceptance_execution_authorized"])
        self.assertFalse(receipt["decision"]["formal_transfer_authorized"])
        self.assertFalse(receipt["decision"]["quality_metric_authorized"])

    def test_gate_is_read_only_and_does_not_import_torch_or_load_model(self) -> None:
        source = (REPO_ROOT / "scripts/llama2_validate_cage_v3_gpu_execution_gate.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("transformers", imported_roots)
        self.assertNotIn("from_pretrained", source)
        self.assertIn("full_file_sha256", source)
        self.assertIn("query_gpu_inventory", source)

    def test_acceptance_runner_is_synthetic_and_has_no_quality_path(self) -> None:
        source = (REPO_ROOT / "scripts/llama2_run_cage_v3_gpu_acceptance.py").read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in ("load_dataset", "tokenizer", "cross_entropy", "perplexity", "wikitext", "pg19"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("synthetic_token_ids", source)
        self.assertIn('"quality_metric_computed": False', source)
        self.assertIn('"real_packed_cuda_cache_allocated": False', source)

    def test_repeat_comparator_excludes_telemetry_and_requires_exact_science(self) -> None:
        source = (REPO_ROOT / "scripts/llama2_compare_cage_v3_gpu_acceptance.py").read_text(encoding="utf-8")
        self.assertIn('left.get("scientific_payload") == right.get("scientific_payload")', source)
        self.assertIn('"telemetry_excluded": True', source)
        self.assertNotIn('left.get("telemetry") == right.get("telemetry")', source)

    def test_gate_validation_rejects_authorization_expansion(self) -> None:
        source_path = REPO_ROOT / "utils/llama2_cage_v3_gpu_execution.py"
        sources = {
            "execution_util": {
                "path": "utils/llama2_cage_v3_gpu_execution.py",
                "sha256": file_sha256(source_path),
            }
        }
        gate = {
            "schema_version": 1,
            "gate_id": "llama2-7b-cage-v3-production-gpu-acceptance-execution-gate-v1",
            "status": "pass",
            "claim_eligible": False,
            "protocol_sha256": "d4e3e0468f0719f00632ddc6e0b40e04cc97c459fa2352cd9fe27cb8585273ba",
            "preflight_receipt_sha256": "63f1473f02a7628ebe677e6e65b7001870514c6f81a630ffb1e3131b155b2d25",
            "sources_manifest_sha256": "5a67eb0aeef944cc2ccda7516e6b88d9ddd803cd0bdabfa1385799aa6d604f4d",
            "checks": {
                "protocol_frozen": True,
                "static_preflight_receipt_frozen_and_passed": True,
                "source_manifest_frozen": True,
                "all_execution_sources_frozen": True,
                "repository_clean": True,
                "single_expected_gpu_with_sufficient_memory": True,
                "production_model_config_identity": True,
                "production_weight_full_sha256_identity": True,
                "weight_identity_matches_static_preflight": True,
                "no_model_weights_loaded": True,
                "no_cuda_computation": True,
                "no_corpus_or_quality_metric": True,
            },
            "frozen_sources": {f"source_{index}": spec for index in range(10) for spec in sources.values()},
            "decision": {
                "gpu_acceptance_execution_authorized": True,
                "formal_transfer_authorized": False,
                "corpus_access_authorized": False,
                "quality_metric_authorized": False,
                "runtime_claims_authorized": False,
            },
        }
        validate_gate_receipt(gate, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(gate)
        mutated["decision"]["quality_metric_authorized"] = True
        with self.assertRaises(Llama2CageV3GPUExecutionError):
            validate_gate_receipt(mutated, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
