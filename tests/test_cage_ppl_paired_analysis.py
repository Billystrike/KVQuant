import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.cage_experiment_config import resolve_method
from utils.cage_ppl import (
    PPL_PAIRED_ANCHOR_INDICES,
    PPL_PAIRED_RAW_METHODS,
    PPL_PROMPT_LENGTHS,
)
from utils.cage_ppl_paired_analysis import (
    PairedPPLAnalysisError,
    aggregate_paired_results,
    load_completed_paired_matrix,
    write_paired_analysis_outputs,
)


def _source_state():
    return {
        "git_commit": "paired-fixture",
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def _methods():
    return [
        resolve_method(method, index)
        for index, method in enumerate(PPL_PAIRED_RAW_METHODS)
    ]


def _records():
    penalties = {
        "fp16": 0.0,
        "kivi-g32-r128": 0.04,
        "kivi-g64-r64": 0.05,
        "cage-r64": 0.06,
        "cage-r128": 0.03,
    }
    records = []
    case_index = 0
    for method in _methods():
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES:
                mean = 1.0 + anchor_index * 0.001 + penalties[method["id"]]
                token_nlls = [mean] * 64
                reference = None
                if method["id"] == "fp16":
                    reference_nlls = list(token_nlls)
                    if case_index < 3:
                        reference_nlls[case_index + 1] -= 0.0305 + case_index * 0.0001
                    deltas = [
                        abs(left - right)
                        for left, right in zip(token_nlls, reference_nlls)
                    ]
                    reference = {
                        "token_nlls": reference_nlls,
                        "mean_absolute_token_nll_delta": sum(deltas) / len(deltas),
                        "max_absolute_token_nll_delta": max(deltas),
                    }
                records.append({
                    "case_id": f"case-{case_index:04d}",
                    "method": {
                        "id": method["id"],
                        "name": method["method"],
                        "resolved_config": method["method_config"],
                    },
                    "input": {
                        "prompt_length": prompt_length,
                        "anchor_index": anchor_index,
                        "continuation_start": 5000 + anchor_index * 5000,
                        "prompt_ids_sha256": f"prompt-{prompt_length}-{anchor_index}",
                        "continuation_ids_sha256": f"continuation-{anchor_index}",
                        "full_ids_sha256": f"full-{prompt_length}-{anchor_index}",
                    },
                    "scoring": {
                        "decode_target_count": 63,
                        "decode_nll_sum": mean * 63,
                        "decode_mean_nll": mean,
                        "token_nlls": token_nlls,
                        "fp16_one_shot_reference": reference,
                    },
                    "provenance": {"source_state": _source_state()},
                })
                case_index += 1
    return records


def _manifest(records):
    return {
        "protocol_stage": "paired_full",
        "methods": _methods(),
        "prompt_lengths": list(PPL_PROMPT_LENGTHS),
        "anchor_indices": list(PPL_PAIRED_ANCHOR_INDICES),
        "source_state": _source_state(),
        "expanded_cases": [
            {"case_id": row["case_id"], "method": row["method"], "input": row["input"]}
            for row in records
        ],
    }


def _quality(tables):
    audit = tables["fp16_calibration_audit"][0]
    return {
        "fp16_calibration_cases": 200,
        "fp16_calibration_max_mean_abs_token_nll_delta": audit[
            "max_case_mean_abs_token_nll_delta"
        ],
        "fp16_calibration_max_abs_token_nll_delta": audit[
            "max_abs_token_nll_delta"
        ],
    }


def _full_quality(tables):
    audit = tables["fp16_calibration_audit"][0]
    return {
        "schema_version": 1,
        "protocol_stage": "paired_full",
        "expected_cases": 1000,
        "completed_cases": 1000,
        "failure_records": 0,
        "completion_gate": "PASS",
        "quality_gate": "NOT_APPLICABLE",
        "fp16_calibration_cases": 200,
        "fp16_calibration_max_mean_abs_token_nll_delta": audit[
            "max_case_mean_abs_token_nll_delta"
        ],
        "fp16_calibration_max_abs_token_nll_delta": audit[
            "max_abs_token_nll_delta"
        ],
        "method_quality": [
            {
                "method_id": row["method_id"],
                "decode_target_count": row["decode_target_count"],
                "decode_nll_sum": row["decode_nll_sum"],
                "decode_mean_nll": row["decode_mean_nll"],
                "decode_perplexity": row["decode_perplexity"],
            }
            for row in tables["method_summary"]
        ],
    }


class PairedPPLAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _records()
        cls.manifest = _manifest(cls.records)
        cls.tables = aggregate_paired_results(cls.records, cls.manifest)

    def test_builds_pre_registered_tables_and_anchor_clustered_pairs(self):
        self.assertEqual(len(self.tables["method_summary"]), 5)
        self.assertEqual(len(self.tables["length_summary"]), 20)
        self.assertEqual(len(self.tables["paired_comparisons"]), 10)
        self.assertEqual(len(self.tables["anchor_deltas"]), 400)
        self.assertEqual(len(self.tables["fp16_calibration_audit"]), 1)
        self.assertEqual(len(self.tables["fp16_calibration_violations"]), 3)

        audit = self.tables["fp16_calibration_audit"][0]
        self.assertEqual(audit["mean_gate"], "PASS")
        self.assertEqual(audit["max_token_gate"], "FAIL")
        self.assertEqual(audit["overall_gate"], "FAIL")
        self.assertEqual(audit["protocol_status"], "RECORDED_DEVIATION")
        self.assertEqual(audit["token_violation_count"], 3)
        self.assertEqual(audit["primary_token_violation_count"], 3)

        r128 = next(
            row for row in self.tables["paired_comparisons"]
            if row["comparison_id"] == "cage-r128_vs_kivi-g32-r128"
            and row["prompt_length"] == 512
        )
        self.assertEqual(r128["anchor_count"], 50)
        self.assertEqual(r128["case_pair_count"], 50)
        self.assertAlmostEqual(r128["mean_paired_delta_nll"], -0.01)
        self.assertEqual(r128["candidate_better_anchors"], 50)
        self.assertAlmostEqual(r128["bootstrap_mean_delta_nll_ci95_low"], -0.01)
        self.assertAlmostEqual(r128["bootstrap_mean_delta_nll_ci95_high"], -0.01)

        r64_overall = next(
            row for row in self.tables["paired_comparisons"]
            if row["comparison_id"] == "cage-r64_vs_kivi-g64-r64"
            and row["scope"] == "overall_diagnostic_anchor_clustered"
        )
        self.assertEqual(r64_overall["anchor_count"], 50)
        self.assertEqual(r64_overall["case_pair_count"], 200)
        self.assertAlmostEqual(r64_overall["mean_paired_delta_nll"], 0.01)
        self.assertEqual(r64_overall["candidate_worse_anchors"], 50)

    def test_bootstrap_outputs_are_bitwise_deterministic(self):
        second = aggregate_paired_results(self.records, self.manifest)
        self.assertEqual(
            self.tables["paired_comparisons"],
            second["paired_comparisons"],
        )

    def test_writes_fourteen_table_and_protocol_outputs_without_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            outputs = write_paired_analysis_outputs(
                output,
                self.tables,
                resolved_manifest=self.manifest,
                quality_summary=_quality(self.tables),
                make_plots=False,
            )
            self.assertEqual(len(outputs), 14)
            self.assertEqual(
                len((output / "paired_comparisons.jsonl").read_text().splitlines()),
                10,
            )
            self.assertEqual(
                len((output / "anchor_deltas.jsonl").read_text().splitlines()),
                400,
            )
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["schema_version"], 2)
            self.assertEqual(protocol["bootstrap"]["seed"], 20260725)
            self.assertEqual(protocol["bootstrap"]["resamples"], 10_000)
            self.assertEqual(protocol["fp16_calibration"]["overall_gate"], "FAIL")
            self.assertEqual(
                protocol["post_run_protocol_record"]["status"],
                "RECORDED_DEVIATION",
            )
            self.assertEqual(
                protocol["post_run_protocol_record"]["acceptance_repeat_result"],
                "PASS_BITWISE",
            )
            self.assertEqual(
                protocol["post_run_protocol_record"][
                    "acceptance_repeat_bitwise_equal_numeric_comparisons"
                ],
                985,
            )
            self.assertEqual(
                len((output / "fp16_calibration_violations.jsonl")
                    .read_text().splitlines()),
                3,
            )
            summary = (output / "paired_ppl_summary.md").read_text()
            self.assertIn("recorded protocol deviation", summary)
            self.assertIn("frozen threshold is not retrospectively changed", summary)

            with self.assertRaisesRegex(PairedPPLAnalysisError, "not empty"):
                write_paired_analysis_outputs(
                    output,
                    self.tables,
                    resolved_manifest=self.manifest,
                    quality_summary=_quality(self.tables),
                    make_plots=False,
                )

    def test_strict_loader_validates_one_thousand_expected_cases(self):
        by_id = {row["case_id"]: row for row in self.records}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases").mkdir()
            (root / "failures").mkdir()
            (root / "summary").mkdir()
            (root / "manifest.resolved.json").write_text(
                json.dumps(self.manifest), encoding="utf-8"
            )
            (root / "summary" / "quality.json").write_text(
                json.dumps(_full_quality(self.tables)), encoding="utf-8"
            )
            with (root / "summary" / "cases.jsonl").open("w", encoding="utf-8") as handle:
                for row in self.records:
                    handle.write(json.dumps(row) + "\n")
            with (root / "summary" / "cases.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id"])
                writer.writeheader()
                for row in self.records:
                    writer.writerow({"case_id": row["case_id"]})
            for row in self.records:
                (root / "cases" / f"{row['case_id']}.json").write_text(
                    "{}", encoding="utf-8"
                )

            with mock.patch(
                "utils.cage_ppl_paired_analysis.validate_completed_ppl_case",
                side_effect=lambda _root, case_id: by_id[case_id],
            ):
                resolved, records, quality = load_completed_paired_matrix(root)
            self.assertEqual(resolved, self.manifest)
            self.assertEqual(len(records), 1000)
            self.assertEqual(quality["completion_gate"], "PASS")


if __name__ == "__main__":
    unittest.main()
