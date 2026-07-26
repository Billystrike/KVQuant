import json
import tempfile
import unittest
from pathlib import Path

from utils.cage_paper_evidence import (
    COMPARISONS,
    PaperEvidenceError,
    build_paper_evidence,
    load_evidence_inputs,
    write_paper_evidence_outputs,
)


METHODS = (
    "fp16", "kivi-g32-r32", "kivi-g32-r64", "kivi-g32-r128",
    "kivi-g64-r64", "kivi-g64-r128", "kivi-g128-r128",
    "cage-r32", "cage-r64", "cage-r128",
)
LENGTHS = (512, 1024, 2048, 4095)


def _state(commit):
    return {
        "git_commit": commit,
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture(root: Path):
    pareto = root / "pareto"
    paired = root / "paired"
    passkey = root / "passkey"
    for path in (pareto, paired, passkey):
        path.mkdir()

    pareto_protocol = {
        "schema_version": 2,
        "aggregate_point_count": 40,
        "primary_error_metric": "joint_post_o_proj_mse",
        "memory_axis": "memory.paper_estimate.total_bytes",
        "source_states": [_state("pareto")],
    }
    _write_json(pareto / "analysis_protocol.json", pareto_protocol)
    pareto_rows = []
    for order, method in enumerate(METHODS):
        for length in LENGTHS:
            baseline_bytes = length * 1000
            if method == "kivi-g32-r128":
                paper_bytes, error = baseline_bytes, 1.0
            elif method == "cage-r128":
                paper_bytes, error = baseline_bytes * 0.95, 1.1
            elif method == "kivi-g64-r64":
                paper_bytes, error = baseline_bytes * 0.9, 1.2
            elif method == "cage-r64":
                paper_bytes, error = baseline_bytes * 0.85, 1.0
            else:
                paper_bytes, error = baseline_bytes * (2.0 - order * 0.05), 2.0 + order
            pareto_rows.append({
                "config_id": method,
                "method": "fp16" if method == "fp16" else method.split("-")[0],
                "prompt_length": length,
                "paper_total_bytes": int(paper_bytes),
                "paper_total_mib": paper_bytes / 1024**2,
                "compression_ratio_vs_fp16": 2.0,
                "primary_error": error,
                "primary_error_sample_pstdev": 0.1,
                "is_pareto_global": True,
            })
    _write_jsonl(pareto / "aggregate_points.jsonl", pareto_rows)

    paired_protocol = {
        "schema_version": 2,
        "case_count": 1000,
        "primary_analysis_scope": (
            "pre_registered_cage_vs_kivi_paired_comparisons_only"
        ),
        "source_state": _state("paired"),
        "post_run_protocol_record": {"status": "RECORDED_DEVIATION"},
    }
    _write_json(paired / "analysis_protocol.json", paired_protocol)
    calibration = {
        "overall_gate": "FAIL",
        "protocol_status": "RECORDED_DEVIATION",
    }
    _write_jsonl(paired / "fp16_calibration_audit.jsonl", [calibration])
    pair_rows = []
    for comparison in COMPARISONS:
        improvement = comparison["candidate_method_id"] == "cage-r64"
        for length in (None, 512, 1024, 2048, 4032):
            delta = -0.01 if improvement else 0.001
            pair_rows.append({
                **comparison,
                "scope": (
                    "overall_diagnostic_anchor_clustered"
                    if length is None else "prompt_length"
                ),
                "prompt_length": length,
                "mean_paired_delta_nll": delta,
                "candidate_ppl_ratio_to_baseline": 0.99 if improvement else 1.001,
                "bootstrap_mean_delta_nll_ci95_low": (
                    -0.02 if improvement else -0.005
                ),
                "bootstrap_mean_delta_nll_ci95_high": (
                    -0.001 if improvement else 0.006
                ),
                "candidate_better_anchors": 35 if improvement else 25,
                "candidate_worse_anchors": 15 if improvement else 25,
            })
    _write_jsonl(paired / "paired_comparisons.jsonl", pair_rows)

    passkey_protocol = {
        "schema_version": 1,
        "protocol_stage": "stage_b",
        "case_count": 300,
        "source_state": _state("passkey"),
    }
    _write_json(passkey / "analysis_protocol.json", passkey_protocol)
    passkey_ids = ("fp16", "kivi-g32-r32", "kivi-g64-r64", "cage-r32", "cage-r64")
    method_rows = []
    for method in passkey_ids:
        exact = 59 if method == "cage-r64" else 60
        method_rows.append({
            "method_id": method,
            "trials": 60,
            "exact_matches": exact,
            "exact_accuracy": exact / 60,
            "contains_matches": 60,
            "contains_accuracy": 1.0,
        })
    _write_jsonl(passkey / "method_summary.jsonl", method_rows)
    pair_rows = []
    for left_index, left in enumerate(passkey_ids):
        for right in passkey_ids[left_index + 1:]:
            candidate_pair = left == "kivi-g64-r64" and right == "cage-r64"
            pair_rows.append({
                "method_a_id": left,
                "method_b_id": right,
                "paired_trials": 60,
                "both_exact": 59 if candidate_pair else 60,
                "method_a_only_exact": 1 if candidate_pair else 0,
                "method_b_only_exact": 0,
                "neither_exact": 0,
            })
    _write_jsonl(passkey / "paired_comparisons.jsonl", pair_rows)
    return pareto, paired, passkey


class PaperEvidenceTests(unittest.TestCase):
    def test_loads_builds_exact_joins_and_preserves_native_context_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            pareto, paired, passkey = _fixture(Path(directory))
            inputs = load_evidence_inputs(pareto, paired, passkey)
            tables = build_paper_evidence(inputs)
            self.assertEqual(len(tables["operating_point_evidence"]), 8)
            self.assertEqual(len(tables["comparison_summary"]), 2)
            exact = [
                row for row in tables["operating_point_evidence"]
                if row["ppl_join_status"] == "exact_prompt_length"
            ]
            unmatched = [
                row for row in tables["operating_point_evidence"]
                if row["ppl_join_status"] == "unmatched_4095_vs_4032"
            ]
            self.assertEqual(len(exact), 6)
            self.assertEqual(len(unmatched), 2)
            self.assertTrue(all(row["ppl_prompt_length"] is None for row in unmatched))

            by_id = {
                row["comparison_id"]: row for row in tables["comparison_summary"]
            }
            self.assertEqual(
                by_id["cage-r128_vs_kivi-g32-r128"]["ppl_status"],
                "descriptive_parity",
            )
            self.assertEqual(
                by_id["cage-r128_vs_kivi-g32-r128"]["passkey_join_status"],
                "method_not_in_stage_b",
            )
            r64 = by_id["cage-r64_vs_kivi-g64-r64"]
            self.assertEqual(r64["ppl_status"], "descriptive_improvement")
            self.assertEqual(r64["passkey_join_status"], "exact_methods")
            self.assertAlmostEqual(r64["candidate_passkey_exact_accuracy_delta"], -1 / 60)
            self.assertEqual(r64["passkey_baseline_only_exact"], 1)

    def test_writes_six_nonplot_outputs_and_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pareto, paired, passkey = _fixture(root)
            inputs = load_evidence_inputs(pareto, paired, passkey)
            tables = build_paper_evidence(inputs)
            output = root / "evidence"
            paths = write_paper_evidence_outputs(
                output, inputs, tables, make_plots=False,
            )
            self.assertEqual(len(paths), 6)
            protocol = json.loads((output / "evidence_protocol.json").read_text())
            self.assertEqual(protocol["schema_version"], 1)
            self.assertEqual(protocol["exact_join_lengths"], [512, 1024, 2048])
            self.assertEqual(
                protocol["unmatched_native_context_lengths"]["policy"],
                "retain both as unmatched; never silently equate them",
            )
            ledger = (output / "claim_ledger.md").read_text()
            self.assertIn("Mechanism ablations", ledger)
            with self.assertRaisesRegex(PaperEvidenceError, "not empty"):
                write_paper_evidence_outputs(output, inputs, tables, make_plots=False)

    def test_rejects_unrecorded_paired_ppl_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            pareto, paired, passkey = _fixture(Path(directory))
            protocol_path = paired / "analysis_protocol.json"
            protocol = json.loads(protocol_path.read_text())
            protocol["post_run_protocol_record"]["status"] = "CONFORMING"
            _write_json(protocol_path, protocol)
            with self.assertRaisesRegex(PaperEvidenceError, "deviation"):
                load_evidence_inputs(pareto, paired, passkey)


if __name__ == "__main__":
    unittest.main()
