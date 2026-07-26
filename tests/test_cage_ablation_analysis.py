import json
import tempfile
import unittest
from pathlib import Path

from utils.cage_ablation_analysis import (
    AblationAnalysisError,
    aggregate_ablation_results,
    write_ablation_analysis_outputs,
)
from utils.cage_experiment_config import load_and_resolve_manifest
from utils.cage_experiment_schema import BYTE_FIELDS, METRIC_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]


def _memory(total, cache_type):
    row = {field: 0 for field in BYTE_FIELDS}
    row["total_bytes"] = total
    row["cache_type"] = cache_type
    return row


def _role(method_id):
    if method_id.startswith("kivi-"):
        return "kivi"
    return method_id.split("-", 2)[2]


def _fixture():
    manifest = load_and_resolve_manifest(REPO_ROOT / "configs" / "cage_ablation_llama2_7b.json")
    penalties = {
        "full": 0.80,
        "k-adaptive": 1.00,
        "v-adaptive": 1.20,
        "uniform": 1.40,
        "fixed-random": 1.30,
        "kivi": 0.90,
    }
    memory_by_role = {
        "full": 100,
        "k-adaptive": 95,
        "v-adaptive": 98,
        "uniform": 90,
        "fixed-random": 100,
        "kivi": 85,
    }
    runs = []
    for method in manifest["methods"]:
        role = _role(method["id"])
        residual = method["method_config"]["residual_length"]
        for length in manifest["prompt_lengths"]:
            for sample_index, sample in enumerate(manifest["sample_ids"]):
                error = penalties[role] + residual / 1000 + length / 1_000_000 + sample_index * 0.01
                total = memory_by_role[role] * 1024**2 + residual * 1024 + length
                metrics = {}
                for metric in METRIC_NAMES:
                    value = 0.5 if metric == "topk_attention_overlap" else error
                    metrics[metric] = {"mean": value, "median": value, "max": value}
                run_id = f"{method['id']}-{length}-{sample}"
                runs.append({
                    "run_id": run_id,
                    "method": {
                        "name": method["method"],
                        "resolved_config": method["method_config"],
                    },
                    "input": {"sample_id": sample, "prompt_length": length},
                    "memory": {
                        "paper_estimate": _memory(total, "paper"),
                        "runtime_tensors": _memory(total * 2, "runtime"),
                    },
                    "metrics_aggregate": metrics,
                    "provenance": {
                        "source_state": {
                            "git_commit": "fixture",
                            "dirty": False,
                            "dirty_sha256": None,
                            "untracked_paths": [],
                        }
                    },
                })
    return manifest, runs


class AblationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.runs = _fixture()
        cls.tables = aggregate_ablation_results(cls.runs, cls.manifest)

    def test_builds_frozen_tables_and_same_budget_random_control(self):
        self.assertEqual(len(self.tables["aggregate_points"]), 48)
        self.assertEqual(len(self.tables["sample_contrasts"]), 96)
        self.assertEqual(len(self.tables["paired_contrasts"]), 32)
        self.assertEqual(len(self.tables["factorial_decomposition"]), 8)
        self.assertEqual(len(self.tables["advancement_decisions"]), 4)
        self.assertEqual(len(self.tables["external_baselines"]), 8)
        random_rows = [
            row for row in self.tables["paired_contrasts"]
            if row["contrast_id"] == "full_vs_fixed-random"
        ]
        self.assertEqual(len(random_rows), 8)
        self.assertTrue(all(row["same_paper_and_runtime_budget"] for row in random_rows))
        self.assertTrue(all(row["full_better_samples"] == 3 for row in random_rows))

    def test_factorial_effects_and_advancement_have_declared_direction(self):
        row = self.tables["factorial_decomposition"][0]
        self.assertGreater(row["full_error_reduction_vs_uniform_percent"], 0)
        self.assertGreater(row["key_only_error_reduction_vs_uniform_percent"], 0)
        self.assertGreater(row["value_only_error_reduction_vs_uniform_percent"], 0)
        self.assertLess(row["key_effect_uniform_context"], 0)
        self.assertLess(row["value_effect_uniform_context"], 0)
        self.assertTrue(all(
            row["not_worse_than_uniform_length_count"] == 4
            and row["advance_to_ppl_ablation"]
            for row in self.tables["advancement_decisions"]
        ))

    def test_writes_fourteen_non_plot_outputs_and_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            paths = write_ablation_analysis_outputs(
                output,
                self.tables,
                resolved_manifest=self.manifest,
                run_count=144,
                make_plots=False,
            )
            self.assertEqual(len(paths), 14)
            self.assertEqual(len((output / "aggregate_points.jsonl").read_text().splitlines()), 48)
            self.assertEqual(len((output / "sample_contrasts.jsonl").read_text().splitlines()), 96)
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["paired_contrast_count"], 32)
            self.assertIn("Same-budget", (output / "ablation_summary.md").read_text())
            with self.assertRaisesRegex(AblationAnalysisError, "not empty"):
                write_ablation_analysis_outputs(
                    output,
                    self.tables,
                    resolved_manifest=self.manifest,
                    run_count=144,
                    make_plots=False,
                )

    def test_rejects_protocol_method_order_change(self):
        broken = dict(self.manifest)
        broken["methods"] = list(reversed(self.manifest["methods"]))
        with self.assertRaisesRegex(AblationAnalysisError, "method IDs/order"):
            aggregate_ablation_results(self.runs, broken)


if __name__ == "__main__":
    unittest.main()
