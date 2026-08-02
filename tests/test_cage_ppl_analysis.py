import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.cage_experiment_config import resolve_method
from utils.cage_ppl import (
    PPL_ANCHOR_INDICES,
    PPL_FULL_RAW_METHODS,
    PPL_PROMPT_LENGTHS,
    PPL_SCHEMA_VERSION,
)
from utils.cage_ppl_analysis import (
    PPLAnalysisError,
    aggregate_ppl_results,
    build_joint_selection,
    load_completed_ppl_matrix,
    load_pareto_analysis,
    write_ppl_analysis_outputs,
)


def source_state():
    return {
        "git_commit": "analysis-fixture",
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def resolved_methods():
    return [resolve_method(method, index) for index, method in enumerate(PPL_FULL_RAW_METHODS)]


def fixture_records():
    records = []
    case_index = 0
    for method_order, method in enumerate(resolved_methods()):
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_ANCHOR_INDICES:
                mean_nll = 1.0 + anchor_index * 0.01 + method_order * 0.005
                case_id = f"case-{case_index:03d}"
                input_record = {
                    "prompt_length": prompt_length,
                    "anchor_index": anchor_index,
                    "continuation_start": 10000 + anchor_index,
                    "prompt_ids_sha256": f"prompt-{prompt_length}-{anchor_index}",
                    "continuation_ids_sha256": f"continuation-{anchor_index}",
                    "full_ids_sha256": f"full-{prompt_length}-{anchor_index}",
                }
                case_method = {
                    "id": method["id"],
                    "name": method["method"],
                    "resolved_config": method["method_config"],
                }
                records.append({
                    "case_id": case_id,
                    "method": case_method,
                    "input": input_record,
                    "scoring": {
                        "decode_target_count": 63,
                        "decode_nll_sum": mean_nll * 63,
                        "decode_mean_nll": mean_nll,
                    },
                    "provenance": {"source_state": source_state()},
                })
                case_index += 1
    return records


def fixture_manifest(records):
    return {
        "protocol_stage": "full",
        "prompt_lengths": list(PPL_PROMPT_LENGTHS),
        "anchor_indices": list(PPL_ANCHOR_INDICES),
        "methods": resolved_methods(),
        "source_state": source_state(),
        "expanded_cases": [
            {"case_id": record["case_id"], "method": record["method"], "input": record["input"]}
            for record in records
        ],
    }


def fixture_quality(tables):
    methods = []
    for row in tables["method_summary"]:
        methods.append({
            "method_id": row["method_id"],
            "completed_cases": row["case_count"],
            "decode_target_count": row["decode_target_count"],
            "decode_nll_sum": row["decode_nll_sum"],
            "decode_mean_nll": row["decode_mean_nll"],
            "decode_perplexity": row["decode_perplexity"],
        })
    lengths = []
    for row in tables["length_summary"]:
        lengths.append({
            "method_id": row["method_id"],
            "prompt_length": row["prompt_length"],
            "completed_cases": row["case_count"],
            "decode_target_count": row["decode_target_count"],
            "decode_nll_sum": row["decode_nll_sum"],
            "decode_mean_nll": row["decode_mean_nll"],
            "decode_perplexity": row["decode_perplexity"],
        })
    return {
        "schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "full",
        "expected_cases": 200,
        "completed_cases": 200,
        "failure_records": 0,
        "completion_gate": "PASS",
        "quality_gate": "NOT_APPLICABLE",
        "method_quality": methods,
        "length_quality": lengths,
        "fp16_calibration_cases": 20,
        "fp16_calibration_max_mean_abs_token_nll_delta": 0.001,
        "fp16_calibration_max_abs_token_nll_delta": 0.01,
    }


def fixture_pareto_rows():
    rows = []
    methods = resolved_methods()
    for prompt_length in (512, 1024, 2048, 4095):
        for order, method in enumerate(methods):
            rows.append({
                "config_id": method["id"],
                "method": method["method"],
                "prompt_length": prompt_length,
                "paper_total_bytes": (10 - order) * prompt_length,
                "paper_total_mib": (10 - order) * prompt_length / 1024**2,
                "primary_error": float(order),
                "primary_error_sample_pstdev": 0.1,
                "compression_ratio_vs_fp16": 1.0 + order,
                "is_pareto_global": True,
            })
    return rows


class PPLAnalysisAggregationTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture_records()
        self.manifest = fixture_manifest(self.records)

    def test_builds_token_weighted_method_length_anchor_and_paired_tables(self):
        tables = aggregate_ppl_results(self.records, self.manifest)
        self.assertEqual(len(tables["method_summary"]), 10)
        self.assertEqual(len(tables["length_summary"]), 40)
        self.assertEqual(len(tables["anchor_summary"]), 50)
        self.assertEqual(len(tables["paired_vs_fp16"]), 45)

        fp16 = tables["method_summary"][0]
        candidate = tables["method_summary"][1]
        self.assertEqual(fp16["case_count"], 20)
        self.assertEqual(fp16["decode_target_count"], 1260)
        self.assertAlmostEqual(fp16["ppl_ratio_vs_fp16"], 1.0)
        self.assertAlmostEqual(candidate["delta_mean_nll_vs_fp16"], 0.005)
        self.assertAlmostEqual(candidate["ppl_ratio_vs_fp16"], math.exp(0.005))

        paired = next(
            row for row in tables["paired_vs_fp16"]
            if row["method_id"] == "kivi-g32-r32" and row["prompt_length"] == 512
        )
        self.assertEqual(paired["pair_count"], 5)
        self.assertEqual(paired["candidate_worse_pairs"], 5)
        self.assertAlmostEqual(paired["mean_paired_delta_nll"], 0.005)

    def test_joint_table_uses_exact_lengths_and_does_not_remap_max_context(self):
        tables = aggregate_ppl_results(self.records, self.manifest)
        joint = build_joint_selection(tables["length_summary"], fixture_pareto_rows())
        self.assertEqual(len(joint), 40)
        self.assertEqual(sum(row["join_status"] == "exact_prompt_length" for row in joint), 30)
        max_context = [row for row in joint if row["ppl_prompt_length"] == 4032]
        self.assertTrue(all(row["pareto_prompt_length"] is None for row in max_context))
        self.assertTrue(all(row["is_memory_ppl_pareto"] is None for row in max_context))

    def test_rejects_method_dependent_input_hashes(self):
        self.records[-1]["input"] = dict(self.records[-1]["input"])
        self.records[-1]["input"]["prompt_ids_sha256"] = "drift"
        from utils import cage_ppl_analysis
        with self.assertRaisesRegex(PPLAnalysisError, "identical PPL inputs"):
            cage_ppl_analysis._validate_coverage(self.records)


class PPLAnalysisArtifactTests(unittest.TestCase):
    def test_strict_loader_checks_artifacts_summaries_and_quality(self):
        records = fixture_records()
        manifest = fixture_manifest(records)
        tables = aggregate_ppl_results(records, manifest)
        quality = fixture_quality(tables)
        by_id = {record["case_id"]: record for record in records}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases").mkdir()
            (root / "failures").mkdir()
            (root / "summary").mkdir()
            (root / "manifest.resolved.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "summary" / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
            with (root / "summary" / "cases.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            with (root / "summary" / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id"])
                writer.writeheader()
                for record in records:
                    writer.writerow({"case_id": record["case_id"]})
            for record in records:
                (root / "cases" / f"{record['case_id']}.json").write_text("{}", encoding="utf-8")

            with mock.patch(
                "utils.cage_ppl_analysis.validate_completed_ppl_case",
                side_effect=lambda _root, case_id: by_id[case_id],
            ):
                loaded_manifest, loaded_records, loaded_quality = load_completed_ppl_matrix(root)
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(len(loaded_records), 200)
            self.assertEqual(loaded_quality["completion_gate"], "PASS")

    def test_loads_frozen_pareto_analysis_and_writes_tables_without_plots(self):
        records = fixture_records()
        manifest = fixture_manifest(records)
        tables = aggregate_ppl_results(records, manifest)
        quality = fixture_quality(tables)
        pareto_rows = fixture_pareto_rows()
        pareto_protocol = {"aggregate_point_count": 40, "schema_version": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pareto = root / "pareto"
            pareto.mkdir()
            (pareto / "analysis_protocol.json").write_text(
                json.dumps(pareto_protocol), encoding="utf-8"
            )
            with (pareto / "aggregate_points.jsonl").open("w", encoding="utf-8") as handle:
                for row in pareto_rows:
                    handle.write(json.dumps(row) + "\n")
            loaded_protocol, loaded_rows = load_pareto_analysis(pareto)
            self.assertEqual(loaded_protocol, pareto_protocol)
            self.assertEqual(len(loaded_rows), 40)

            output = root / "analysis"
            outputs = write_ppl_analysis_outputs(
                output,
                tables,
                resolved_manifest=manifest,
                quality_summary=quality,
                pareto_protocol=loaded_protocol,
                pareto_rows=loaded_rows,
                make_plots=False,
            )
            self.assertEqual(len(outputs), 12)
            self.assertEqual(len((output / "method_summary.jsonl").read_text().splitlines()), 10)
            self.assertEqual(len((output / "joint_selection.jsonl").read_text().splitlines()), 40)
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["pareto_join"]["matched_rows"], 30)
            self.assertEqual(protocol["pareto_join"]["unmatched_rows"], 10)

            with self.assertRaisesRegex(PPLAnalysisError, "not empty"):
                write_ppl_analysis_outputs(
                    output,
                    tables,
                    resolved_manifest=manifest,
                    quality_summary=quality,
                    make_plots=False,
                )


if __name__ == "__main__":
    unittest.main()
