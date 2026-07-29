import json
import tempfile
import unittest
from pathlib import Path

from utils.cage_experiment_config import resolve_method
from utils.cage_ppl import (
    PPL_MECHANISM_ABLATION_RAW_METHODS,
    PPL_PAIRED_ANCHOR_INDICES,
    PPL_PROMPT_LENGTHS,
)
from utils.cage_ppl_mechanism_analysis import (
    MechanismPPLAnalysisError,
    aggregate_mechanism_results,
    write_mechanism_analysis_outputs,
)


def _source_state():
    return {
        "git_commit": "mechanism-fixture",
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def _methods():
    return [
        resolve_method(method, index)
        for index, method in enumerate(PPL_MECHANISM_ABLATION_RAW_METHODS)
    ]


def _role(method_id):
    return method_id.split("-", 2)[2]


def _records():
    penalties = {
        "full": 0.00,
        "k-adaptive": 0.01,
        "v-adaptive": 0.04,
        "uniform": 0.05,
        "fixed-random": 0.20,
    }
    records = []
    case_index = 0
    for method in _methods():
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES:
                mean = (
                    1.0 + penalties[_role(method["id"])]
                    + anchor_index * 0.001 + prompt_length * 0.000001
                )
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
                        "decode_perplexity": pow(2.718281828459045, mean),
                    },
                    "provenance": {"source_state": _source_state()},
                })
                case_index += 1
    return records


def _manifest(records):
    return {
        "protocol_stage": "mechanism_ablation_full",
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
    return {
        "quality_gate": "NOT_APPLICABLE",
    }


class MechanismPPLAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _records()
        cls.manifest = _manifest(cls.records)
        cls.tables = aggregate_mechanism_results(cls.records, cls.manifest)

    def test_builds_pre_registered_paired_and_factorial_tables(self):
        self.assertEqual(len(self.tables["method_summary"]), 10)
        self.assertEqual(len(self.tables["length_summary"]), 40)
        self.assertEqual(len(self.tables["paired_comparisons"]), 40)
        self.assertEqual(len(self.tables["anchor_deltas"]), 1600)
        self.assertEqual(len(self.tables["factorial_summary"]), 10)
        self.assertEqual(len(self.tables["factorial_anchor_effects"]), 400)

        random_row = next(
            row for row in self.tables["paired_comparisons"]
            if row["comparison_id"] == "r64_full_vs_fixed-random"
            and row["scope"] == "overall_anchor_clustered"
        )
        self.assertAlmostEqual(random_row["mean_paired_delta_nll"], -0.20)
        self.assertEqual(random_row["candidate_better_anchors"], 50)
        self.assertAlmostEqual(random_row["bootstrap_mean_delta_nll_ci95_low"], -0.20)
        self.assertAlmostEqual(random_row["bootstrap_mean_delta_nll_ci95_high"], -0.20)

        factorial = next(
            row for row in self.tables["factorial_summary"]
            if row["residual_length"] == 64
            and row["scope"] == "overall_anchor_clustered"
        )
        self.assertAlmostEqual(factorial["mean_full_minus_uniform"], -0.05)
        self.assertAlmostEqual(factorial["mean_key_adaptive_minus_uniform"], -0.04)
        self.assertAlmostEqual(factorial["mean_value_adaptive_minus_uniform"], -0.01)
        self.assertAlmostEqual(
            factorial["mean_interaction_full_minus_key_minus_value_plus_uniform"], 0.0
        )

    def test_bootstrap_outputs_are_deterministic(self):
        second = aggregate_mechanism_results(self.records, self.manifest)
        self.assertEqual(
            self.tables["paired_comparisons"], second["paired_comparisons"]
        )
        self.assertEqual(self.tables["factorial_summary"], second["factorial_summary"])

    def test_writes_fourteen_non_plot_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            outputs = write_mechanism_analysis_outputs(
                output,
                self.tables,
                resolved_manifest=self.manifest,
                quality_summary=_quality(self.tables),
                make_plots=False,
            )
            self.assertEqual(len(outputs), 14)
            self.assertEqual(
                len((output / "paired_comparisons.jsonl").read_text().splitlines()), 40
            )
            self.assertEqual(
                len((output / "anchor_deltas.jsonl").read_text().splitlines()), 1600
            )
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["bootstrap"]["seed"], 20260726)
            self.assertEqual(protocol["bootstrap"]["resamples"], 10_000)
            self.assertIn("Overall paired contrasts", (output / "ppl_mechanism_summary.md").read_text())
            with self.assertRaisesRegex(MechanismPPLAnalysisError, "not empty"):
                write_mechanism_analysis_outputs(
                    output,
                    self.tables,
                    resolved_manifest=self.manifest,
                    quality_summary=_quality(self.tables),
                    make_plots=False,
                )

    def test_rejects_non_frozen_method_order(self):
        broken = dict(self.manifest)
        broken["methods"] = list(reversed(self.manifest["methods"]))
        with self.assertRaisesRegex(MechanismPPLAnalysisError, "methods differ"):
            aggregate_mechanism_results(self.records, broken)


if __name__ == "__main__":
    unittest.main()
