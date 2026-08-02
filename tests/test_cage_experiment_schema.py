import copy
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from utils.cage_experiment_io import atomic_write_json, atomic_write_jsonl
from utils.cage_experiment_schema import (
    BYTE_FIELDS,
    METRIC_NAMES,
    is_valid_completed_point,
    validate_completed_point,
)


def _byte_summary(seed):
    summary = {name: 0 for name in BYTE_FIELDS}
    base_fields = BYTE_FIELDS[:8]
    for offset, name in enumerate(base_fields):
        summary[name] = seed + offset
    summary["payload_only_bytes"] = (
        summary["key_payload_bytes"] + summary["value_payload_bytes"]
    )
    summary["metadata_bytes"] = sum(summary[name] for name in (
        "key_scale_bytes", "value_scale_bytes",
        "key_min_or_zp_bytes", "value_min_or_zp_bytes",
    ))
    summary["total_bytes"] = sum(summary[name] for name in base_fields)
    summary["cache_type"] = "fp16"
    return summary


def _sum_byte_summaries(summaries):
    result = {name: sum(summary[name] for summary in summaries) for name in BYTE_FIELDS}
    result["cache_type"] = "model_total"
    return result


def _metric_aggregates(layers):
    return {
        name: {
            "mean": float(statistics.fmean(row[name] for row in layers)),
            "median": float(statistics.median(row[name] for row in layers)),
            "max": float(max(row[name] for row in layers)),
        }
        for name in METRIC_NAMES
    }


def completed_artifacts(run_id="run", layer_count=2, prompt_length=5):
    layers = []
    for index in range(layer_count):
        layers.append({
            "schema_version": 1,
            "run_id": run_id,
            "layer_index": index,
            "method": "fp16",
            "prompt_length": prompt_length,
            "phase": "teacher_forced_decode",
            "query_source": "fp16_reference_final_position",
            "memory": {
                "paper_estimate": _byte_summary(index + 1),
                "runtime_tensors": _byte_summary((index + 1) * 2),
                "cuda_peak_diagnostic": {
                    "max_allocated_bytes": 0,
                    "max_reserved_bytes": 0,
                },
                "cache_structure": {
                    "key": {
                        "total_tokens": prompt_length,
                        "quantized_history_tokens": 0,
                        "fp16_residual_tokens": prompt_length,
                    },
                    "value": {
                        "total_tokens": prompt_length,
                        "quantized_history_tokens": 0,
                        "fp16_residual_tokens": prompt_length,
                    },
                },
            },
            **{
                name: (1.0 if name == "topk_attention_overlap" else 0.0)
                for name in METRIC_NAMES
            },
        })
    paper_total = _sum_byte_summaries([
        row["memory"]["paper_estimate"] for row in layers
    ])
    runtime_total = _sum_byte_summaries([
        row["memory"]["runtime_tensors"] for row in layers
    ])
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "model": {"reference": "model", "dtype": "float16", "device": "cpu"},
        "method": {"name": "fp16", "resolved_config": {}},
        "input": {
            "sample_id": "sample",
            "text_sha256": "0" * 64,
            "prompt_length": prompt_length,
            "continuation_tokens": 1,
        },
        "quantization": {"method": "fp16"},
        "measurement": {
            "phase": "teacher_forced_decode",
            "query_source": "fp16_reference_final_position",
            "query_count": 1,
            "layer_count": layer_count,
        },
        "memory": {
            "paper_estimate": paper_total,
            "runtime_tensors": runtime_total,
            "cuda_peak_diagnostic": {
                "max_allocated_bytes": 10,
                "max_reserved_bytes": 20,
            },
        },
        "metrics_aggregate": _metric_aggregates(layers),
        "runtime_diagnostics": {"load_seconds": 0.1},
        "provenance": {"source_state": {"git_commit": "abc"}, "deterministic_seed": 7},
    }
    return run, layers


class CageExperimentSchemaTest(unittest.TestCase):
    def _write(self, root, run, layers):
        atomic_write_json(root / "runs" / f"{run['run_id']}.json", run)
        atomic_write_jsonl(root / "layers" / f"{run['run_id']}.jsonl", layers)

    def test_valid_completed_point_passes_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, layers = completed_artifacts()
            self._write(root, run, layers)
            self.assertEqual(validate_completed_point(root, "run"), run)
            self.assertTrue(is_valid_completed_point(root, "run"))

    def test_missing_or_corrupt_layer_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, layers = completed_artifacts()
            atomic_write_json(root / "runs" / "run.json", run)
            with self.assertRaisesRegex(ValueError, "layer.*missing"):
                validate_completed_point(root, "run")
            layer_path = root / "layers" / "run.jsonl"
            layer_path.parent.mkdir()
            layer_path.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layer.*JSON"):
                validate_completed_point(root, "run")
            layer_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                validate_completed_point(root, "run")

    def test_layer_count_and_indices_must_be_exact_and_contiguous(self):
        cases = {
            "count": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows[1].update(layer_index=0),
            "gap": lambda rows: rows[1].update(layer_index=2),
        }
        for message, mutate in cases.items():
            with self.subTest(case=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(layers)
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "count|indices"):
                    validate_completed_point(root, "run")

    def test_layer_schema_run_and_context_must_match(self):
        mutations = [
            lambda row: row.update(schema_version=2),
            lambda row: row.update(run_id="other"),
            lambda row: row.update(method="kivi"),
            lambda row: row.update(prompt_length=6),
            lambda row: row.update(phase="prefill"),
            lambda row: row.update(query_source="candidate"),
            lambda row: row.update(extra=True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(layers[0])
                self._write(root, run, layers)
                with self.assertRaises(ValueError):
                    validate_completed_point(root, "run")

    def test_metrics_must_be_complete_finite_real_and_nonnegative(self):
        mutations = [
            lambda row: row.pop(METRIC_NAMES[0]),
            lambda row: row.update({METRIC_NAMES[0]: math.nan}),
            lambda row: row.update({METRIC_NAMES[0]: math.inf}),
            lambda row: row.update({METRIC_NAMES[0]: True}),
            lambda row: row.update({METRIC_NAMES[0]: -1.0}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(layers[0])
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "metric"):
                    validate_completed_point(root, "run")

    def test_topk_attention_overlap_must_be_in_closed_unit_interval(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                layers[0]["topk_attention_overlap"] = value
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "topk_attention_overlap.*\[0, 1\]"):
                    validate_completed_point(root, "run")

    def test_run_topk_aggregate_statistics_must_be_in_closed_unit_interval(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                run["metrics_aggregate"]["topk_attention_overlap"]["mean"] = value
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "topk_attention_overlap.mean.*\[0, 1\]"):
                    validate_completed_point(root, "run")

    def test_run_metric_mean_median_and_max_must_match_layer_rows(self):
        metric = "relative_k_reconstruction_error"
        for statistic_name in ("mean", "median", "max"):
            with self.subTest(statistic=statistic_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                layers[0][metric] = 1.0
                layers[1][metric] = 3.0
                run["metrics_aggregate"] = _metric_aggregates(layers)
                run["metrics_aggregate"][metric][statistic_name] += 0.01
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, f"{metric}.{statistic_name}.*layer"):
                    validate_completed_point(root, "run")

    def test_run_metric_aggregate_names_and_statistics_are_exact(self):
        mutations = [
            lambda aggregate: aggregate.pop(METRIC_NAMES[0]),
            lambda aggregate: aggregate.update(extra_metric={
                "mean": 0.0, "median": 0.0, "max": 0.0,
            }),
            lambda aggregate: aggregate[METRIC_NAMES[0]].pop("median"),
            lambda aggregate: aggregate[METRIC_NAMES[0]].update(extra=0.0),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(run["metrics_aggregate"])
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "metrics_aggregate"):
                    validate_completed_point(root, "run")

    def test_layer_and_model_cache_summary_derived_byte_arithmetic_is_exact(self):
        mutations = [
            lambda run, layers: layers[0]["memory"]["paper_estimate"].update(
                payload_only_bytes=999
            ),
            lambda run, layers: layers[0]["memory"]["runtime_tensors"].update(
                metadata_bytes=999
            ),
            lambda run, layers: layers[0]["memory"]["paper_estimate"].update(
                total_bytes=999
            ),
            lambda run, layers: run["memory"]["paper_estimate"].update(
                payload_only_bytes=999
            ),
            lambda run, layers: run["memory"]["runtime_tensors"].update(
                metadata_bytes=999
            ),
            lambda run, layers: run["memory"]["runtime_tensors"].update(
                total_bytes=999
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(run, layers)
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "byte arithmetic"):
                    validate_completed_point(root, "run")

    def test_model_memory_must_equal_recursive_layer_sums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, layers = completed_artifacts()
            summary = run["memory"]["paper_estimate"]
            summary["key_payload_bytes"] += 1
            summary["payload_only_bytes"] += 1
            summary["total_bytes"] += 1
            self._write(root, run, layers)
            with self.assertRaisesRegex(ValueError, "memory.*sum"):
                validate_completed_point(root, "run")

    def test_run_top_level_schema_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, layers = completed_artifacts()
            run["unexpected"] = True
            self._write(root, run, layers)
            with self.assertRaisesRegex(ValueError, "unknown"):
                validate_completed_point(root, "run")

    def test_invalid_cache_structure_is_rejected(self):
        mutations = [
            lambda cache: cache["key"].update(total_tokens=4),
            lambda cache: cache["value"].update(quantized_history_tokens=1),
            lambda cache: cache["key"].update(fp16_residual_tokens=-1),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, layers = completed_artifacts()
                mutate(layers[0]["memory"]["cache_structure"])
                self._write(root, run, layers)
                with self.assertRaisesRegex(ValueError, "cache structure"):
                    validate_completed_point(root, "run")


if __name__ == "__main__":
    unittest.main()
