from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_cpu_acceptance import (
    Llama2CageV3CPUAcceptanceError,
    load_cpu_protocol,
    validate_cpu_protocol,
    validate_static_preflight_payloads,
)
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_cpu_acceptance_protocol_v1.json"


class Llama2CageV3CPUAcceptanceProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_checked_in_cpu_protocol_and_static_receipts_are_frozen(self):
        loaded, digest = load_cpu_protocol(PROTOCOL_PATH, repo_root=REPO_ROOT)
        self.assertEqual(loaded, self.protocol)
        self.assertEqual(digest, "ea5299d08d0dbceabd701fcdd063ec3ab04b8fa5ad4c0ad6573f161c0649f6f1")
        quota, preflight = validate_static_preflight_payloads(loaded, repo_root=REPO_ROOT)
        self.assertFalse(quota["llama2_metrics_consumed"])
        self.assertEqual(preflight["status"], "pass")
        self.assertFalse(preflight["qwen3_promotion_pass"])

    def test_server_to_repository_newline_normalization_is_explicit_and_canonical(self):
        for spec in self.protocol["static_preflight_artifacts"].values():
            path = REPO_ROOT / spec["checked_in_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(file_sha256(path), spec["checked_in_sha256"])
            self.assertEqual(path.stat().st_size, spec["server_size_bytes"] + 1)
            self.assertEqual(canonical_sha256(payload), spec["canonical_sha256"])
            self.assertIn("terminal LF", spec["normalization"])

    def test_failed_cross_platform_hash_attempt_is_preserved_without_scientific_change(self):
        repair = self.protocol["administrative_repair"]
        receipt_path = REPO_ROOT / repair["failed_attempt_receipt_path"]
        self.assertEqual(file_sha256(receipt_path), repair["failed_attempt_receipt_sha256"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["source_commit"], "277cfadc0d03d30e29b7f9dff621673864e3286e")
        self.assertFalse(receipt["test_summary"]["direct_cpu_acceptance_invoked"])
        self.assertFalse(receipt["repair_boundary"]["candidate_algorithm_changed"])
        self.assertFalse(repair["candidate_algorithm_changed"])
        self.assertFalse(repair["quota_plan_changed"])
        self.assertFalse(repair["metric_changed"])
        self.assertFalse(repair["authorization_changed"])

    def test_protocol_rejects_target_metric_or_authorization_mutation(self):
        mutated = copy.deepcopy(self.protocol)
        mutated["transfer_boundary"]["llama2_quality_metrics_consumed"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUAcceptanceError, "quality metrics"):
            validate_cpu_protocol(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_boundary"]["gpu_acceptance_authorized"] = True
        with self.assertRaisesRegex(Llama2CageV3CPUAcceptanceError, "gpu_acceptance_authorized"):
            validate_cpu_protocol(mutated, repo_root=REPO_ROOT)

    def test_protocol_rejects_runtime_source_or_fixture_mutation(self):
        mutated = copy.deepcopy(self.protocol)
        mutated["frozen_sources"]["llama_cage_v3_runtime"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(Llama2CageV3CPUAcceptanceError, "frozen source"):
            validate_cpu_protocol(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.protocol)
        mutated["cpu_fixture"]["two_bit_channels"] = 3
        with self.assertRaisesRegex(Llama2CageV3CPUAcceptanceError, "fixture"):
            validate_cpu_protocol(mutated, repo_root=REPO_ROOT)

    def test_runner_is_cpu_only_and_does_not_load_production_weights(self):
        source = (REPO_ROOT / "scripts" / "llama2_cage_v3_cpu_acceptance.py").read_text(encoding="utf-8")
        self.assertIn("AutoConfig.from_pretrained", source)
        self.assertNotIn("AutoModel", source)
        self.assertNotIn("LlamaForCausalLM.from_pretrained", source)
        self.assertNotIn(".to(\"cuda\")", source)
        self.assertNotIn(".cuda()", source)
        self.assertIn('"full_model_weights_loaded": False', source)
        self.assertIn('"device_used": "cpu"', source)


if __name__ == "__main__":
    unittest.main()
