import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.qwen3_compare_cage_v3_promotion_acceptance import _payload
from utils.qwen3_cage_v3_promotion_acceptance import (
    SCIENTIFIC_FIELDS,
    acceptance_inputs,
    expand_acceptance_cases,
    expand_promotion_methods,
    load_acceptance_execution,
    validate_static_preflight_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = (
    REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_acceptance_execution_v1.json"
)


def _minimal_manifest(execution):
    frozen = execution["acceptance_input"]
    full_window = list(range(4096))
    cases = []
    compact = []
    for length, case_id in zip((1024, 2048, 4032), frozen["input_case_ids"], strict=True):
        identity = {
            "partition": "holdout",
            "document_id": frozen["document_id"],
            "anchor_index": frozen["anchor_index"],
            "prompt_length": length,
        }
        cases.append({"case_id": case_id, "identity": identity})
        compact.append({"case_id": case_id, "prompt_length": length})
    return {
        "anchors": [
            {
                "anchor_id": frozen["anchor_id"],
                "document_id": frozen["document_id"],
                "anchor_index": frozen["anchor_index"],
                "partition": "holdout",
                "full_window_ids": full_window,
                "cases": compact,
            }
        ],
        "cases": cases,
    }


class CageV3PromotionAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.execution, self.execution_sha256, self.protocol, self.protocol_sha256 = (
            load_acceptance_execution(
                EXECUTION_PATH, repo_root=REPO_ROOT, verify_server_artifacts=False
            )
        )

    def test_execution_freezes_acceptance_only_boundary(self):
        self.assertEqual(
            self.execution["authorization"],
            {
                "gpu_acceptance": True,
                "full_holdout": False,
                "holdout_interpretation": False,
                "pg19_test": False,
                "llama2_execution": False,
                "kitty_llama_port": False,
                "paper_claims": False,
                "runtime_claims": False,
            },
        )
        self.assertEqual(self.execution["static_preflight"]["status"], "pass")
        self.assertFalse(self.execution["static_preflight"]["reads_holdout_method_metrics"])

    def test_real_static_preflight_schema_does_not_require_failures_field(self):
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "claim_eligible": False,
            "protocol_sha256": self.protocol_sha256,
            "input_manifest_sha256": self.execution["input_manifest"]["sha256"],
            "acceptance_input": {
                key: copy.deepcopy(self.execution["acceptance_input"][key])
                for key in ("anchor_id", "anchor_index", "document_id", "input_case_ids")
            },
            "full_case_counts": {
                "cage_qwen3": 480,
                "kitty_qwen3": 120,
                "total": 600,
            },
            "boundary": {
                "full_holdout_authorized": False,
                "gpu_acceptance_authorized_by_this_preflight": False,
                "kitty_llama_port_authorized": False,
                "llama2_execution_authorized": False,
                "pg19_test_access_authorized": False,
                "reads_holdout_method_metrics": False,
            },
        }
        self.assertNotIn("failures", receipt)
        validate_static_preflight_receipt(
            receipt,
            execution=self.execution,
            protocol_sha256=self.protocol_sha256,
        )
        receipt["boundary"]["reads_holdout_method_metrics"] = True
        with self.assertRaises(RuntimeError):
            validate_static_preflight_receipt(
                receipt,
                execution=self.execution,
                protocol_sha256=self.protocol_sha256,
            )

    def test_method_expansion_is_exact_and_memory_frozen(self):
        cage = expand_promotion_methods(
            self.protocol, repo_root=REPO_ROOT, partition="cage_qwen3"
        )
        kitty = expand_promotion_methods(
            self.protocol, repo_root=REPO_ROOT, partition="kitty_qwen3"
        )
        self.assertEqual(len(cage), 12)
        self.assertEqual(len(kitty), 3)
        self.assertEqual(
            [row["metric_method_id"] for row in cage[::3]],
            [
                "fp16",
                "kivi-kittypro-matched",
                "cage-v1-kittypro-matched",
                "cage-v3-sr2-sink32-calibrated",
            ],
        )
        frozen = {
            (method["method_id"], point["prompt_length"]): point["packed_bytes"]
            for method in self.protocol["method_grid"]
            for point in method["points"]
        }
        for method in [*cage, *kitty]:
            self.assertEqual(
                method["packed_bytes"],
                frozen[(method["metric_method_id"], method["prompt_length"])],
            )

    def test_acceptance_input_is_frozen_holdout_anchor_with_nested_prompts(self):
        inputs = acceptance_inputs(_minimal_manifest(self.execution), execution=self.execution)
        self.assertEqual(set(inputs), {1024, 2048, 4032})
        self.assertEqual([len(inputs[length]["prompt_ids"]) for length in inputs], [1024, 2048, 4032])
        self.assertEqual(inputs[1024]["continuation_ids"], list(range(4032, 4096)))
        self.assertEqual(inputs[1024]["continuation_ids"], inputs[4032]["continuation_ids"])

    def test_acceptance_input_rejects_partition_anchor_or_case_id_mutation(self):
        for mutation in ("partition", "anchor", "case"):
            manifest = _minimal_manifest(self.execution)
            if mutation == "partition":
                manifest["anchors"][0]["partition"] = "screen"
            elif mutation == "anchor":
                manifest["anchors"][0]["anchor_id"] = "changed"
            else:
                manifest["cases"][0]["case_id"] = "changed"
                manifest["anchors"][0]["cases"][0]["case_id"] = "changed"
            with self.assertRaises(RuntimeError):
                acceptance_inputs(manifest, execution=self.execution)

    def test_expands_exact_partition_counts_and_deterministic_case_ids(self):
        manifest = _minimal_manifest(self.execution)
        first = {}
        for partition, expected in (("cage_qwen3", 12), ("kitty_qwen3", 3)):
            cases = expand_acceptance_cases(
                execution=self.execution,
                execution_sha256=self.execution_sha256,
                protocol=self.protocol,
                manifest=manifest,
                partition=partition,
                repo_root=REPO_ROOT,
            )
            repeated = expand_acceptance_cases(
                execution=self.execution,
                execution_sha256=self.execution_sha256,
                protocol=self.protocol,
                manifest=copy.deepcopy(manifest),
                partition=partition,
                repo_root=REPO_ROOT,
            )
            self.assertEqual(len(cases), expected)
            self.assertEqual(cases, repeated)
            self.assertEqual(len({row["case_id"] for row in cases}), expected)
            first[partition] = cases
        self.assertTrue(all(row["partition"] == "cage_qwen3" for row in first["cage_qwen3"]))

    def test_repeat_payload_excludes_runtime_provenance_and_timestamps(self):
        scientific = {
            "case_id": "case-a",
            "partition": "cage_qwen3",
            "method": {"id": "fp16-l1024"},
            "input": {"prompt_length": 1024},
            "memory": {"model_total_bytes": 1},
            "scoring": {"mean_nll": 2.0},
            "quality_cache": {"reported_seq_length": 1087},
        }
        self.assertEqual(tuple(scientific), SCIENTIFIC_FIELDS)
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("a", "b")]
            for index, root in enumerate(roots):
                (root / "cases").mkdir(parents=True)
                record = {
                    **scientific,
                    "identity": {"repeat": index},
                    "model": {"loaded_at": index},
                    "runtime": {"elapsed_seconds": index + 1.0},
                    "completed_at_utc": f"2026-08-16T00:00:0{index}+00:00",
                }
                (root / "cases" / "case-a.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(_payload(roots[0]), _payload(roots[1]))

    def test_gpu_runner_exposes_no_full_holdout_or_metric_interpretation_stage(self):
        source = (
            REPO_ROOT / "scripts" / "qwen3_run_cage_v3_promotion_acceptance.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("expand_full", source)
        self.assertNotIn("local_perturbation", source)


if __name__ == "__main__":
    unittest.main()
