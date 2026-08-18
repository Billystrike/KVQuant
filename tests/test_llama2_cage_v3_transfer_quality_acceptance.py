import copy
import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_acceptance import (
    Llama2CageV3TransferQualityAcceptanceError,
    expand_acceptance_cases,
    validate_completed_case,
    validate_execution,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_execution_v1.json"
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_protocol_v1.json"
SOURCE_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_sources_v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "llama2_run_cage_v3_transfer_quality_acceptance.py"
COMPARATOR_PATH = REPO_ROOT / "scripts" / "llama2_compare_cage_v3_transfer_quality_acceptance.py"
GATE_PATH = REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_quality_acceptance_gate.py"


class Llama2CageV3TransferQualityAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
        cls.protocol, _ = load_transfer_quality_protocol(PROTOCOL_PATH, repo_root=REPO_ROOT)

    def _minimal_manifest(self):
        records = []
        for length in (1024, 2048, 4032):
            records.append({
                "input_id": f"input-{length}",
                "identity": {
                    "anchor_index": 0,
                    "prompt_length": length,
                    "continuation_start": 10647,
                },
                "prompt_ids": [1] * length,
                "continuation_ids": [2] * 64,
            })
        return {"inputs": records}

    def test_execution_and_all_frozen_sources_are_exact(self):
        validate_execution(self.execution, repo_root=REPO_ROOT)
        source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(source["source_id"], "llama2-7b-cage-v3-transfer-quality-acceptance-sources-v1")
        for spec in source["files"].values():
            self.assertEqual(file_sha256(REPO_ROOT / spec["path"]), spec["sha256"])

    def test_expands_exact_12_acceptance_cases_in_length_then_role_order(self):
        cases = expand_acceptance_cases(
            execution=self.execution,
            execution_sha256=file_sha256(EXECUTION_PATH),
            protocol=self.protocol,
            input_manifest=self._minimal_manifest(),
        )
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["case_id"] for case in cases}), 12)
        self.assertEqual(
            [case["method"]["role"] for case in cases[:4]],
            ["quality_reference", "transfer_candidate", "primary_predecessor", "primary_uniform_baseline"],
        )
        self.assertEqual([cases[index]["input"]["identity"]["prompt_length"] for index in (0, 4, 8)], [1024, 2048, 4032])

    def test_completed_case_requires_all_64_targets_and_final_cache_length(self):
        case = expand_acceptance_cases(
            execution=self.execution,
            execution_sha256=file_sha256(EXECUTION_PATH),
            protocol=self.protocol,
            input_manifest=self._minimal_manifest(),
        )[0]
        record = {
            "schema_version": 1,
            "status": "completed",
            **case,
            "scoring": {
                "primary_metric": "cache_conditioned_all_64_target_mean_nll",
                "boundary_target_count": 1,
                "decode_target_count": 63,
                "all_target_count": 64,
                "token_nlls": [1.0] * 64,
                "nll_sum": 64.0,
                "mean_nll": 1.0,
            },
            "cache": {
                "layer_count": 32,
                "final_cache_length": 1087,
                "all_layer_lengths_equal": True,
                "method_family": "fp16",
                "quantized_path_verified": False,
            },
        }
        validate_completed_case(record, case)
        changed = copy.deepcopy(record)
        changed["scoring"]["token_nlls"] = [1.0] * 63
        with self.assertRaises(Llama2CageV3TransferQualityAcceptanceError):
            validate_completed_case(changed, case)

    def test_rejects_execution_authorization_or_scoring_mutation(self):
        for mutate in (
            lambda value: value["current_boundary"].__setitem__("quality_acceptance_execution_authorized", True),
            lambda value: value["scoring"].__setitem__("primary_target_count_per_case", 63),
            lambda value: value["acceptance"].__setitem__("anchor_indices", [1]),
        ):
            changed = copy.deepcopy(self.execution)
            mutate(changed)
            with self.assertRaises(Llama2CageV3TransferQualityAcceptanceError):
                validate_execution(changed, repo_root=REPO_ROOT)

    def test_runner_comparator_and_gate_preserve_execution_boundaries(self):
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("for index in range(63)", runner)
        self.assertIn('"all_target_count": 64', runner)
        self.assertIn("install_llama_cage_v3_config", runner)
        self.assertIn("resolve_method", runner)
        comparator = COMPARATOR_PATH.read_text(encoding="utf-8")
        self.assertIn("SCIENTIFIC_FIELDS", comparator)
        self.assertIn('"telemetry_excluded": True', comparator)
        gate = GATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained", gate)
        self.assertNotIn("cross_entropy", gate)
        self.assertIn('"no_model_weights_loaded": True', gate)
        self.assertIn('"no_quality_metric_computed_or_read": True', gate)


if __name__ == "__main__":
    unittest.main()
