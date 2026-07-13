import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_cage_experiment_schema import completed_artifacts
from utils.cage_experiment_io import atomic_write_json, atomic_write_jsonl
from utils.cage_experiment_schema import ExperimentPointError, validate_completed_point


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

    def test_point_id_uses_only_point_local_scientific_identity(self):
        sample = types.SimpleNamespace(sample_id="doc-001", text="same scientific input")
        source = {"git_commit": "abc", "dirty": False, "dirty_sha256": None}
        base = {
            "job_id": "acceptance-cage-r32",
            "method": "cage",
            "model": {
                "reference": r"C:\models\llama",
                "dtype": "float16",
                "device": "cuda",
                "max_position_embeddings": 4096,
            },
            "method_config": {"k_bits": 2, "v_bits": 2, "residual_length": 32},
            "sample_ids": ["doc-001"],
            "prompt_lengths": [512, 2048],
            "measurement": {"decode_tokens": 1, "seed": 7},
            "prompts_file": r"C:\acceptance\prompts.jsonl",
            "output_dir": r"C:\results",
        }
        full = json.loads(json.dumps(base))
        full.update({
            "job_id": "full-cage-r32",
            "sample_ids": ["doc-001", "doc-002", "doc-003"],
            "prompt_lengths": [512, 1024, 2048, 4095],
            "prompts_file": r"D:\full\prompts.jsonl",
            "output_dir": r"D:\elsewhere",
        })

        acceptance_id = self.worker._point_id(base, sample, 512, source)
        full_id = self.worker._point_id(full, sample, 512, source)
        self.assertEqual(acceptance_id, full_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, layers = completed_artifacts(full_id, layer_count=1, prompt_length=512)
            atomic_write_json(root / "runs" / f"{full_id}.json", run)
            atomic_write_jsonl(root / "layers" / f"{full_id}.jsonl", layers)
            self.assertTrue(self.worker.completed_run_exists(root, full_id))

        scientific_mutations = [
            lambda job, point, state: job["model"].update(reference="other-model"),
            lambda job, point, state: job.update(method="kivi"),
            lambda job, point, state: job["method_config"].update(residual_length=64),
            lambda job, point, state: job["measurement"].update(seed=8),
            lambda job, point, state: setattr(point, "sample_id", "doc-002"),
            lambda job, point, state: setattr(point, "text", "changed text"),
            lambda job, point, state: state.update(git_commit="def"),
        ]
        for mutate in scientific_mutations:
            with self.subTest(mutation=mutate):
                job = json.loads(json.dumps(base))
                point = types.SimpleNamespace(sample_id=sample.sample_id, text=sample.text)
                state = dict(source)
                mutate(job, point, state)
                self.assertNotEqual(
                    acceptance_id,
                    self.worker._point_id(job, point, 512, state),
                )
        self.assertNotEqual(acceptance_id, self.worker._point_id(base, sample, 1024, source))

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
            root = Path(directory)
            run, layers = completed_artifacts("abc")
            atomic_write_json(root / "runs" / "abc.json", run)
            atomic_write_jsonl(root / "layers" / "abc.jsonl", layers)
            self.assertTrue(self.worker.completed_run_exists(root, "abc"))
            (root / "layers" / "abc.jsonl").write_text("corrupt", encoding="utf-8")
            self.assertFalse(self.worker.completed_run_exists(root, "abc"))

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
                 mock.patch.object(self.worker, "_load_model",
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
            run_record, layer_records = completed_artifacts(
                "completed", layer_count=1, prompt_length=2
            )
            atomic_write_json(root / "out" / "runs" / "completed.json", run_record)
            atomic_write_jsonl(root / "out" / "layers" / "completed.jsonl", layer_records)
            failure = root / "out" / "failures" / "completed.json"
            failure.parent.mkdir(parents=True)
            failure.write_text("{}", encoding="utf-8")
            with mock.patch.object(self.worker, "_point_id", return_value="completed"), \
                 mock.patch.object(self.worker, "_load_model",
                                   side_effect=AssertionError("weights must not load")):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 0)
            self.assertFalse(failure.exists())

    def test_invalid_completed_artifacts_are_rerun_and_atomically_overwritten(self):
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "prompts_file": "prompts.jsonl", "sample_ids": ["s"], "prompt_lengths": [2],
            "methods": [{"id": "full-fp", "method": "fp16", "method_config": {}}],
            "measurement": {"decode_tokens": 1, "seed": 7}, "output_dir": "out",
        }

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(model_type="llama")
                self.model = types.SimpleNamespace(layers=[types.SimpleNamespace()])

            def __call__(self, *, input_ids, use_cache, return_dict, past_key_values=None):
                del return_dict
                output = types.SimpleNamespace(logits=self_tensor.zeros(1, 1, 2))
                if use_cache and past_key_values is None:
                    key = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                    value = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                    output.past_key_values = ((key, value),)
                return output

        self_tensor = self.worker.torch
        source = {"git_commit": "abc", "dirty": False, "dirty_sha256": None}
        provenance = {"source_state": source, "deterministic_seed": 7}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            text = "hello"
            (root / "prompts.jsonl").write_text(
                json.dumps({"sample_id": "s", "text": text}) + "\n", encoding="utf-8"
            )
            sample = types.SimpleNamespace(sample_id="s", text=text)
            job = manifest.copy()
            job = self.worker.expand_jobs(manifest)[0]
            run_id = self.worker._point_id(job, sample, 2, source)
            atomic_write_json(root / "out" / "runs" / f"{run_id}.json", {
                "run_id": run_id, "status": "completed"
            })
            corrupt_layer = root / "out" / "layers" / f"{run_id}.jsonl"
            corrupt_layer.parent.mkdir(parents=True)
            corrupt_layer.write_text("corrupt\n", encoding="utf-8")
            prepared = {
                ("s", 2): {
                    "prompt_ids": self_tensor.tensor([[1, 2]]),
                    "continuation_ids": self_tensor.tensor([[3]]),
                    "text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                }
            }
            with mock.patch.object(self.worker, "collect_provenance", return_value=provenance), \
                 mock.patch.object(self.worker, "_prepare_inputs", return_value=(object(), prepared)), \
                 mock.patch.object(self.worker, "_load_model", return_value=(FakeModel(), 0.1)):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 0)
            self.assertEqual(validate_completed_point(root / "out", run_id)["run_id"], run_id)

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

    def test_worker_rejects_non_finite_model_output_and_cache_tensors(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ExperimentPointError, "non-finite"
            ):
                self.worker.require_finite_tensor_tree(
                    "point output", {"cache": (self.worker.torch.tensor([value]),)}
                )

    def test_cuda_peak_diagnostic_is_updated_separately_from_prefill_cache_summary(self):
        memory = {"paper_estimate": {"total_bytes": 10}}
        with mock.patch.object(self.worker.torch.cuda, "max_memory_allocated", return_value=30), \
             mock.patch.object(self.worker.torch.cuda, "max_memory_reserved", return_value=40):
            self.worker.update_point_cuda_diagnostic(memory, "cuda")
        self.assertEqual(memory["paper_estimate"]["total_bytes"], 10)
        self.assertEqual(memory["cuda_peak_diagnostic"], {
            "max_allocated_bytes": 30,
            "max_reserved_bytes": 40,
        })

    def test_success_removes_stale_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory) / "failures" / "run.json"
            failure.parent.mkdir()
            failure.write_text("{}", encoding="utf-8")
            self.worker.remove_stale_failure(Path(directory), "run")
            self.assertFalse(failure.exists())

    def test_failure_classifier_aligns_stage_category_code_and_retryability(self):
        cases = [
            (ValueError("bad prompt"), "input", False,
             ("input_error", 2, False)),
            (self.worker.torch.cuda.OutOfMemoryError("CUDA out of memory"), "load", False,
             ("cuda_out_of_memory", 3, False)),
            (RuntimeError("checkpoint is corrupt"), "load", False,
             ("model_load_error", 4, False)),
            (RuntimeError("reference failed"), "reference", False,
             ("experiment_point_error", 2, False)),
            (RuntimeError("decode failed"), "decode", False,
             ("experiment_point_error", 2, False)),
            (RuntimeError("capture reset failed"), "capture_reset", True,
             ("transient_model_state", 5, True)),
        ]
        for error, stage, unusable, expected in cases:
            with self.subTest(stage=stage):
                classification = self.worker.classify_worker_failure(
                    error, stage, unusable=unusable
                )
                self.assertEqual(
                    (classification.category, classification.exit_code,
                     classification.retryable),
                    expected,
                )
                self.assertEqual(classification.stage, stage)


if __name__ == "__main__":
    unittest.main()
