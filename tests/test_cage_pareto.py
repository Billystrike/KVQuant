import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.cage_experiment_config import expand_jobs
from utils.cage_experiment_schema import BYTE_FIELDS, METRIC_NAMES
from utils.cage_pareto import (
    ParetoAnalysisError,
    aggregate_points,
    build_sample_points,
    build_trends,
    load_completed_matrix,
    write_analysis_outputs,
)


def _manifest(methods, samples=("doc-a", "doc-b"), lengths=(512,)):
    resolved = {
        "model": {
            "reference": "model",
            "dtype": "float16",
            "device": "cuda",
            "max_position_embeddings": 4096,
        },
        "prompts_file": "prompts.jsonl",
        "sample_ids": list(samples),
        "prompt_lengths": list(lengths),
        "methods": methods,
        "measurement": {"decode_tokens": 1, "seed": 0},
        "output_dir": "output",
    }
    resolved["jobs"] = expand_jobs(resolved)
    return resolved


def _memory(total, cache_type):
    summary = {field: 0 for field in BYTE_FIELDS}
    summary["total_bytes"] = total
    summary["cache_type"] = cache_type
    return summary


def _run(run_id, method, config, sample, length, paper_bytes, error):
    aggregates = {}
    for metric in METRIC_NAMES:
        if method == "fp16":
            value = 1.0 if metric == "topk_attention_overlap" else 0.0
        elif metric == "topk_attention_overlap":
            value = 0.5
        else:
            value = float(error)
        aggregates[metric] = {"mean": value, "median": value, "max": value}
    return {
        "run_id": run_id,
        "model": {
            "reference": "model",
            "dtype": "float16",
            "device": "cuda",
            "model_type": "llama",
        },
        "method": {"name": method, "resolved_config": config},
        "input": {"sample_id": sample, "prompt_length": length},
        "measurement": {"layer_count": 32},
        "memory": {
            "paper_estimate": _memory(paper_bytes, f"{method}_paper"),
            "runtime_tensors": _memory(paper_bytes * 2, f"{method}_runtime"),
            "cuda_peak_diagnostic": {
                "max_allocated_bytes": paper_bytes * 10,
                "max_reserved_bytes": paper_bytes * 12,
            },
        },
        "metrics_aggregate": aggregates,
        "provenance": {
            "source_state": {
                "git_commit": "abc",
                "dirty": False,
                "dirty_sha256": None,
                "untracked_paths": [],
            }
        },
    }


class ParetoAggregationTests(unittest.TestCase):
    def setUp(self):
        self.fp16 = {"id": "fp16", "method": "fp16", "method_config": {}}
        self.kivi_config = {
            "k_bits": 2,
            "v_bits": 2,
            "group_size": 32,
            "residual_length": 32,
        }
        self.kivi = {
            "id": "kivi-g32-r32",
            "method": "kivi",
            "method_config": self.kivi_config,
        }

    def test_aggregates_samples_normalizes_memory_and_marks_tradeoff(self):
        manifest = _manifest([self.fp16, self.kivi])
        runs = [
            _run("f-a", "fp16", {}, "doc-a", 512, 100, 0),
            _run("f-b", "fp16", {}, "doc-b", 512, 100, 0),
            _run("k-a", "kivi", self.kivi_config, "doc-a", 512, 50, 1),
            _run("k-b", "kivi", self.kivi_config, "doc-b", 512, 50, 3),
        ]

        points = aggregate_points(runs, manifest)
        self.assertEqual(len(points), 2)
        kivi = next(point for point in points if point["method"] == "kivi")
        self.assertEqual(kivi["sample_count"], 2)
        self.assertEqual(kivi["primary_error"], 2.0)
        self.assertEqual(kivi["primary_error_sample_pstdev"], 1.0)
        self.assertEqual(kivi["compression_ratio_vs_fp16"], 2.0)
        self.assertEqual(kivi["memory_fraction_of_fp16"], 0.5)
        self.assertTrue(kivi["is_pareto_global"])
        self.assertEqual(kivi["pareto_sample_count"], 2)
        self.assertTrue(kivi["is_pareto_all_samples"])
        self.assertTrue(next(point for point in points if point["method"] == "fp16")["is_pareto_global"])

        sample_points = build_sample_points(points)
        self.assertEqual(len(sample_points), 4)
        self.assertTrue(all(point["is_pareto_global"] for point in sample_points))

    def test_rejects_sample_dependent_cache_memory(self):
        manifest = _manifest([self.fp16, self.kivi])
        runs = [
            _run("f-a", "fp16", {}, "doc-a", 512, 100, 0),
            _run("f-b", "fp16", {}, "doc-b", 512, 100, 0),
            _run("k-a", "kivi", self.kivi_config, "doc-a", 512, 50, 1),
            _run("k-b", "kivi", self.kivi_config, "doc-b", 512, 51, 1),
        ]
        with self.assertRaisesRegex(ParetoAnalysisError, "sample-dependent paper_estimate"):
            aggregate_points(runs, manifest)

    def test_rejects_missing_sample_coverage(self):
        manifest = _manifest([self.fp16, self.kivi])
        runs = [
            _run("f-a", "fp16", {}, "doc-a", 512, 100, 0),
            _run("f-b", "fp16", {}, "doc-b", 512, 100, 0),
            _run("k-a", "kivi", self.kivi_config, "doc-a", 512, 50, 1),
        ]
        with self.assertRaisesRegex(ParetoAnalysisError, "sample coverage differs"):
            aggregate_points(runs, manifest)

    def test_marks_strictly_dominated_configuration(self):
        cage_config = {"residual_length": 32}
        cage = {"id": "cage-r32", "method": "cage", "method_config": cage_config}
        manifest = _manifest([self.fp16, self.kivi, cage])
        runs = []
        for sample in ("doc-a", "doc-b"):
            runs.extend([
                _run(f"f-{sample}", "fp16", {}, sample, 512, 100, 0),
                _run(f"k-{sample}", "kivi", self.kivi_config, sample, 512, 50, 2),
                _run(f"c-{sample}", "cage", cage_config, sample, 512, 60, 3),
            ])
        points = aggregate_points(runs, manifest)
        cage_point = next(point for point in points if point["method"] == "cage")
        self.assertFalse(cage_point["is_pareto_global"])
        self.assertEqual(cage_point["dominated_by_count_global"], 1)
        self.assertTrue(cage_point["is_pareto_within_method"])

    def test_reports_when_mean_dominance_is_not_stable_for_every_sample(self):
        cage_config = {"residual_length": 32}
        cage = {"id": "cage-r32", "method": "cage", "method_config": cage_config}
        manifest = _manifest([self.fp16, self.kivi, cage])
        runs = [
            _run("f-a", "fp16", {}, "doc-a", 512, 100, 0),
            _run("f-b", "fp16", {}, "doc-b", 512, 100, 0),
            _run("k-a", "kivi", self.kivi_config, "doc-a", 512, 50, 1),
            _run("k-b", "kivi", self.kivi_config, "doc-b", 512, 50, 4),
            _run("c-a", "cage", cage_config, "doc-a", 512, 60, 2),
            _run("c-b", "cage", cage_config, "doc-b", 512, 60, 3),
        ]

        points = aggregate_points(runs, manifest)
        cage_point = next(point for point in points if point["method"] == "cage")
        self.assertFalse(cage_point["is_pareto_global"])
        self.assertEqual(cage_point["pareto_sample_count"], 1)
        self.assertEqual(json.loads(cage_point["pareto_sample_ids_json"]), ["doc-b"])

        sample_points = build_sample_points(points)
        cage_by_sample = {
            point["sample_id"]: point
            for point in sample_points
            if point["method"] == "cage"
        }
        self.assertFalse(cage_by_sample["doc-a"]["is_pareto_global"])
        self.assertTrue(cage_by_sample["doc-b"]["is_pareto_global"])

    def test_builds_length_trends_without_cross_length_pareto(self):
        manifest = _manifest([self.fp16, self.kivi], lengths=(512, 1024))
        runs = []
        for sample in ("doc-a", "doc-b"):
            for length, fp16_memory, kivi_memory, error in (
                (512, 100, 50, 1),
                (1024, 200, 100, 2),
            ):
                runs.extend([
                    _run(f"f-{sample}-{length}", "fp16", {}, sample, length, fp16_memory, 0),
                    _run(
                        f"k-{sample}-{length}", "kivi", self.kivi_config,
                        sample, length, kivi_memory, error,
                    ),
                ])
        trends = build_trends(aggregate_points(runs, manifest))
        kivi_1024 = next(
            row for row in trends
            if row["config_id"] == "kivi-g32-r32" and row["prompt_length"] == 1024
        )
        self.assertEqual(kivi_1024["previous_prompt_length"], 512)
        self.assertEqual(kivi_1024["paper_growth_ratio_vs_previous"], 2.0)
        self.assertEqual(kivi_1024["primary_error_delta_vs_previous"], 1.0)
        self.assertEqual(kivi_1024["primary_error_ratio_vs_previous"], 2.0)

    def test_writes_tables_protocol_and_markdown_without_plots(self):
        manifest = _manifest([self.fp16, self.kivi])
        runs = [
            _run("f-a", "fp16", {}, "doc-a", 512, 100, 0),
            _run("f-b", "fp16", {}, "doc-b", 512, 100, 0),
            _run("k-a", "kivi", self.kivi_config, "doc-a", 512, 50, 1),
            _run("k-b", "kivi", self.kivi_config, "doc-b", 512, 50, 1),
        ]
        points = aggregate_points(runs, manifest)
        trends = build_trends(points)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            paths = write_analysis_outputs(
                output,
                points,
                trends,
                resolved_manifest=manifest,
                run_count=4,
                make_plots=False,
            )
            self.assertEqual(len(paths), 10)
            self.assertTrue((output / "aggregate_points.csv").is_file())
            self.assertTrue((output / "pareto_points.jsonl").is_file())
            self.assertTrue((output / "sample_points.csv").is_file())
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["primary_error_metric"], "joint_post_o_proj_mse")
            self.assertEqual(protocol["sample_point_count"], 4)
            self.assertIn("Prompt length 512", (output / "pareto_summary.md").read_text())


class CompletedMatrixLoadingTests(unittest.TestCase):
    def test_loads_summary_ids_without_recomputing_current_source_identity(self):
        manifest = _manifest([{"id": "fp16", "method": "fp16", "method_config": {}}], samples=("doc",))
        run = _run("frozen-id", "fp16", {}, "doc", 512, 100, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("runs", "layers", "failures", "summary"):
                (root / name).mkdir()
            (root / "manifest.resolved.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "summary" / "runs.jsonl").write_text(
                json.dumps({"run_id": "frozen-id", "status": "completed"}) + "\n",
                encoding="utf-8",
            )
            (root / "runs" / "frozen-id.json").write_text("{}", encoding="utf-8")
            (root / "layers" / "frozen-id.jsonl").write_text("{}\n", encoding="utf-8")
            with mock.patch("utils.cage_pareto.validate_completed_point", return_value=run) as validate:
                resolved, runs = load_completed_matrix(root)
            self.assertEqual(resolved["sample_ids"], ["doc"])
            self.assertEqual([item["run_id"] for item in runs], ["frozen-id"])
            validate.assert_called_once_with(root, "frozen-id")


if __name__ == "__main__":
    unittest.main()
