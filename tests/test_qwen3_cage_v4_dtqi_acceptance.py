import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.qwen3_compare_cage_v4_dtqi_acceptance import compare_repeats
from utils.qwen3_cage_v4_dtqi_acceptance import (
    SCIENTIFIC_FIELDS,
    expand_acceptance_cases,
    load_acceptance_execution,
    load_json,
    validate_cpu_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = (
    REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_gpu_acceptance_v1.json"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "configs"
    / "qwen3_8b_cage_v4_dtqi_cpu_acceptance_receipt_v1.json"
)


def _minimal_manifest(metric_protocol):
    document_id = metric_protocol["acceptance_policy"]["document_id"]
    full_window = list(range(4096))
    cases = []
    compact = []
    for length in (1024, 2048, 4032):
        case_id = f"case-{length}"
        identity = {
            "partition": "screen",
            "document_id": document_id,
            "anchor_index": 0,
            "prompt_length": length,
        }
        cases.append({"case_id": case_id, "identity": identity})
        compact.append({"case_id": case_id, "prompt_length": length})
    return {
        "anchors": [
            {
                "document_id": document_id,
                "anchor_index": 0,
                "partition": "screen",
                "full_window_ids": full_window,
                "cases": compact,
            }
        ],
        "cases": cases,
    }


def _scientific_record(case_id="case-a", mean_nll=1.0):
    return {
        "case_id": case_id,
        "method": {"id": "cage-v4-dtqi-l1024"},
        "input": {"prompt_length": 1024},
        "memory": {"model_total_bytes": 45849600},
        "scoring": {"mean_nll": mean_nll},
        "cache": {"reported_seq_length": 1088},
        "resume": {"length_after": 1088},
    }


class CageV4DTQIAcceptanceTest(unittest.TestCase):
    def test_checked_in_cpu_receipt_and_execution_are_frozen(self):
        receipt = load_json(RECEIPT_PATH)
        validate_cpu_receipt(receipt, verify_artifacts=False)
        execution, execution_sha256, protocol, metric, quota = (
            load_acceptance_execution(
                EXECUTION_PATH,
                repo_root=REPO_ROOT,
                verify_artifacts=False,
            )
        )
        self.assertEqual(len(execution_sha256), 64)
        self.assertEqual(execution["acceptance"]["case_count_per_repeat"], 3)
        self.assertEqual(protocol["candidate"]["points"][0]["recent_query_window"], 176)
        self.assertEqual(metric["acceptance_policy"]["anchor_index"], 0)
        self.assertEqual(
            len(
                [
                    row
                    for row in quota["plans"]
                    if row["family_id"] == "pure-sr2-sink32-uniform32"
                ]
            ),
            3,
        )

    def test_expands_exact_three_nested_acceptance_cases(self):
        execution, execution_sha256, protocol, metric, quota = (
            load_acceptance_execution(
                EXECUTION_PATH,
                repo_root=REPO_ROOT,
                verify_artifacts=False,
            )
        )
        cases = expand_acceptance_cases(
            execution=execution,
            execution_sha256=execution_sha256,
            dtqi_protocol=protocol,
            metric_protocol=metric,
            quota_plan=quota,
            input_manifest=_minimal_manifest(metric),
        )
        self.assertEqual([case["input"]["prompt_length"] for case in cases], [1024, 2048, 4032])
        self.assertEqual([len(case["prompt_ids"]) for case in cases], [1024, 2048, 4032])
        self.assertEqual(cases[0]["prompt_ids"], list(range(3008, 4032)))
        self.assertEqual(cases[1]["prompt_ids"], list(range(1984, 4032)))
        self.assertEqual(cases[2]["prompt_ids"], list(range(4032)))
        self.assertTrue(all(case["continuation_ids"] == list(range(4032, 4096)) for case in cases))
        self.assertEqual([case["method"]["config"]["recent_query_window"] for case in cases], [176, 288, 112])
        self.assertEqual([case["method"]["config"]["residual_length"] for case in cases], [176, 288, 112])
        self.assertTrue(all(sum(case["method"]["config"]["two_bit_channels"]) == 1152 for case in cases))

    def test_cpu_receipt_mutation_is_rejected(self):
        receipt = load_json(RECEIPT_PATH)
        receipt = copy.deepcopy(receipt)
        receipt["authorization"]["gpu_full_screen"] = True
        with self.assertRaisesRegex(RuntimeError, "authorization changed"):
            validate_cpu_receipt(receipt, verify_artifacts=False)

    def test_execution_boundary_blocks_full_screen_holdout_and_test(self):
        execution = load_json(EXECUTION_PATH)
        boundary = execution["execution_boundary"]
        self.assertTrue(boundary["gpu_acceptance_authorized"])
        self.assertFalse(boundary["gpu_full_screen_authorized"])
        self.assertFalse(boundary["holdout_method_metrics_authorized"])
        self.assertFalse(boundary["pg19_test_access_authorized"])
        source = (REPO_ROOT / "scripts" / "qwen3_run_cage_v4_dtqi_acceptance.py").read_text(encoding="utf-8")
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("acceptance_full", source)

    def test_repeat_comparator_is_bitwise_and_excludes_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("a", "b")]
            identity = {"execution_sha256": "a" * 64}
            model = {"parameter_count": 8190735360}
            for index, root in enumerate(roots):
                (root / "cases").mkdir(parents=True)
                summary = {
                    "schema_version": 1,
                    "status": "pass",
                    "claim_eligible": False,
                    "stage": "gpu_acceptance",
                    "expected_cases": 3,
                    "completed_cases": 3,
                    "new_cases": 3,
                    "resumed_cases": 0,
                    "failure_records": 0,
                    "identity": identity,
                    "model": model,
                    "full_screen_authorized": False,
                    "holdout_accessed": False,
                    "pg19_test_accessed": False,
                }
                (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                for case_index in range(3):
                    record = _scientific_record(f"case-{case_index}")
                    record.update(
                        status="completed",
                        runtime={"elapsed_seconds": index + 0.1},
                        completed_at_utc=f"2026-08-13T00:00:0{index}+00:00",
                    )
                    (root / "cases" / f"case-{case_index}.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )
            report = compare_repeats(*roots)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["fields"], list(SCIENTIFIC_FIELDS))
            changed = load_json(roots[1] / "cases" / "case-1.json")
            changed["scoring"]["mean_nll"] = 1.0000000000000002
            (roots[1] / "cases" / "case-1.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            report = compare_repeats(*roots)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["mismatch_case_ids"], ["case-1"])


if __name__ == "__main__":
    unittest.main()
