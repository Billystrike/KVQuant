import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "cage_experiment_worker.py"


def _load():
    spec = importlib.util.spec_from_file_location("cage_experiment_worker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CageExperimentWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = _load()

    def test_parse_args_requires_manifest_and_nonnegative_job_index(self):
        args = self.worker.parse_args(["--manifest", "resolved.json", "--job-index", "2"])
        self.assertEqual((args.manifest, args.job_index), ("resolved.json", 2))
        with self.assertRaises(SystemExit):
            self.worker.parse_args(["--manifest", "x", "--job-index", "-1"])

    def test_apply_method_config_sets_all_values_before_loading(self):
        config = types.SimpleNamespace(model_type="llama")
        values = {"k_bits": 2, "v_bits": 4, "group_size": 32, "residual_length": 64}
        self.worker.apply_method_config(config, "kivi", values)
        self.assertFalse(config.cage_enable)
        for key, value in values.items():
            self.assertEqual(getattr(config, key), value)
        cage = types.SimpleNamespace(model_type="llama")
        cage_values = {"k_bits": 2, "v_bits": 2, "residual_length": 32,
                       "cage_mode": "fake", "cage_k_enable": True}
        self.worker.apply_method_config(cage, "cage", cage_values)
        self.assertTrue(cage.cage_enable)
        self.assertEqual(cage.cage_mode, "fake")
        self.assertTrue(cage.use_flash)
        self.assertEqual(cage.group_size, 32)

    def test_aggregate_layer_metrics_ignores_layer_index(self):
        result = self.worker.aggregate_layer_metrics([
            {"layer_index": 0, "prompt_length": 8, "method": "cage",
             "relative_k_reconstruction_error": 1.0},
            {"layer_index": 1, "prompt_length": 8, "method": "cage",
             "relative_k_reconstruction_error": 3.0},
        ])
        self.assertEqual(result, {"relative_k_reconstruction_error":
                                  {"mean": 2.0, "median": 2.0, "max": 3.0}})

    def test_fp16_zero_records_have_identical_metric_schema(self):
        schema = self.worker.METRIC_NAMES
        records = self.worker.fp16_zero_layer_records(2)
        self.assertEqual([record["layer_index"] for record in records], [0, 1])
        self.assertEqual(tuple(records[0].keys())[3:], schema)
        self.assertTrue(all(value == 0.0 for record in records for value in list(record.values())[3:]))

    def test_completed_run_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs" / "abc.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            self.assertTrue(self.worker.completed_run_exists(Path(directory), "abc"))
            path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            self.assertFalse(self.worker.completed_run_exists(Path(directory), "abc"))

    def test_run_job_skips_weights_when_all_points_are_completed(self):
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "prompts_file": "prompts.jsonl", "sample_ids": ["s"], "prompt_lengths": [2],
            "methods": [{"id": "fp", "method": "fp16", "method_config": {}}],
            "measurement": {"decode_tokens": 1, "seed": 1}, "output_dir": "out",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prompts.jsonl").write_text(json.dumps({"sample_id": "s", "text": "hello"}) + "\n",
                                                encoding="utf-8")
            with mock.patch.object(self.worker, "completed_run_exists", return_value=True), \
                 mock.patch.object(self.worker, "_load_inputs_and_model",
                                   side_effect=AssertionError("weights must not load")):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 0)

    def test_completed_run_skip_removes_stale_failure_without_loading_weights(self):
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "prompts_file": "prompts.jsonl", "sample_ids": ["s"], "prompt_lengths": [2],
            "methods": [{"id": "fp", "method": "fp16", "method_config": {}}],
            "measurement": {"decode_tokens": 1, "seed": 1}, "output_dir": "out",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prompts.jsonl").write_text(
                json.dumps({"sample_id": "s", "text": "hello"}) + "\n", encoding="utf-8"
            )
            run = root / "out" / "runs" / "completed.json"
            run.parent.mkdir(parents=True)
            run.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            failure = root / "out" / "failures" / "completed.json"
            failure.parent.mkdir(parents=True)
            failure.write_text("{}", encoding="utf-8")
            with mock.patch.object(self.worker, "_point_id", return_value="completed"), \
                 mock.patch.object(self.worker, "_load_inputs_and_model",
                                   side_effect=AssertionError("weights must not load")):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 0)
            self.assertFalse(failure.exists())

    def test_run_job_rejects_negative_job_index(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with mock.patch.object(self.worker, "_load_manifest", return_value={}), \
                 mock.patch.object(self.worker, "expand_jobs", return_value=[{"sentinel": True}]):
                with self.assertRaisesRegex(ValueError, "job-index -1 out of range"):
                    self.worker.run_job(manifest, -1)

    def test_validate_layer_counts_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, "layer metric count 1.*layer memory count 2"):
            self.worker.attach_layer_context([{"layer_index": 0}], [{}, {}], "run", "cage", 8)

    def test_release_reference_output_precedes_peak_reset_and_prefill(self):
        events = []
        reference = object()
        released = self.worker.release_reference_output(reference, lambda: events.append("release"))
        self.assertIsNone(released)
        events.append("reset_peak")
        events.append("prefill")
        self.assertEqual(events, ["release", "reset_peak", "prefill"])

    def test_success_removes_stale_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory) / "failures" / "run.json"
            failure.parent.mkdir()
            failure.write_text("{}", encoding="utf-8")
            self.worker.remove_stale_failure(Path(directory), "run")
            self.assertFalse(failure.exists())

    def test_failure_classifier_maps_load_oom_and_transient_without_code_one(self):
        load_error = RuntimeError("checkpoint is corrupt")
        oom = self.worker.torch.cuda.OutOfMemoryError("CUDA out of memory")
        transient = RuntimeError("capture reset failed")

        self.assertEqual(self.worker.classify_worker_failure(load_error, "load"), 4)
        self.assertEqual(self.worker.classify_worker_failure(oom, "prefill"), 3)
        self.assertEqual(
            self.worker.classify_worker_failure(transient, "capture_reset", unusable=True), 5
        )
        possible_codes = {
            self.worker.classify_worker_failure(error, stage, unusable=unusable)
            for error, stage, unusable in (
                (ValueError("bad manifest"), "input", False),
                (load_error, "load", False),
                (oom, "decode", False),
                (transient, "runtime", False),
                (transient, "capture_reset", True),
            )
        }
        self.assertNotIn(1, possible_codes)
        self.assertEqual(possible_codes, {2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
