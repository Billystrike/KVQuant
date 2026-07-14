import importlib.util
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_cage_experiment_schema import completed_artifacts
import utils.cage_experiment_io as cage_io
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

    def _resolved_cage_manifest(self, output_dir):
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "prompts_file": str(Path(output_dir).parent / "prompts.jsonl"),
            "sample_ids": ["s"],
            "prompt_lengths": [2],
            "methods": [{
                "id": "cage-r2",
                "method": "cage",
                "method_config": {
                    "k_bits": 2, "v_bits": 2, "residual_length": 2,
                    "cage_mode": "fake", "cage_k_enable": True,
                    "cage_v_enable": True, "cage_k_importance": "q2_var",
                    "cage_k_group_sizes": [2], "cage_k_clip_percentiles": [1.0],
                    "cage_k_num_buckets": 1, "cage_v_importance": "wo_var",
                    "cage_v_group_sizes": [2], "cage_v_clip_percentiles": [1.0],
                    "cage_v_num_buckets": 1,
                },
            }],
            "measurement": {"decode_tokens": 1, "seed": 7},
            "output_dir": str(output_dir),
        }
        manifest["jobs"] = self.worker.expand_jobs(manifest)
        return manifest

    def test_worker_strictly_validates_resolved_cage_method_config(self):
        mutations = [
            lambda config: config.update(k_bits=4),
            lambda config: config.update(v_bits=4),
            lambda config: config.update(cage_k_enable=False),
            lambda config: config.update(cage_v_enable=False),
            lambda config: config.update(cage_k_importance="variance"),
            lambda config: config.update(cage_v_importance="variance"),
            lambda config: config.pop("cage_k_num_buckets"),
            lambda config: config.update(unknown=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self._resolved_cage_manifest(root / "out")
            path = root / "manifest.resolved.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(self.worker._load_manifest(path)["methods"], valid["methods"])
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    invalid = copy.deepcopy(valid)
                    mutate(invalid["methods"][0]["method_config"])
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.worker._load_manifest(path)

    def test_invalid_resolved_manifest_exits_two_before_tokenizer_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._resolved_cage_manifest(root / "out")
            manifest["methods"][0]["method_config"]["k_bits"] = 4
            path = root / "manifest.resolved.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prompts.jsonl").write_text(
                json.dumps({"sample_id": "s", "text": "long enough"}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(self.worker, "_prepare_inputs") as prepare, \
                 mock.patch.object(self.worker, "_load_model") as load:
                self.assertEqual(
                    self.worker.main(["--manifest", str(path), "--job-index", "0"]),
                    2,
                )
            prepare.assert_not_called()
            load.assert_not_called()

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

        acceptance_id = cage_io.point_id(base, sample, 512, source)
        full_id = cage_io.point_id(full, sample, 512, source)
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
                    cage_io.point_id(job, point, 512, state),
                )
        self.assertNotEqual(acceptance_id, cage_io.point_id(base, sample, 1024, source))

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

    def test_native_model_context_accepts_exact_unscaled_config(self):
        self.assertTrue(hasattr(self.worker, "validate_native_model_context"))
        config = types.SimpleNamespace(
            max_position_embeddings=4096,
            rope_scaling=None,
        )
        self.worker.validate_native_model_context(
            config, {"max_position_embeddings": 4096}
        )

    def test_native_model_context_rejects_mismatch_and_rope_scaling(self):
        self.assertTrue(hasattr(self.worker, "validate_native_model_context"))
        cases = [
            (
                types.SimpleNamespace(max_position_embeddings=8192, rope_scaling=None),
                "max_position_embeddings.*8192.*4096",
            ),
            (
                types.SimpleNamespace(
                    max_position_embeddings=4096,
                    rope_scaling={"rope_type": "linear", "factor": 2.0},
                ),
                "rope_scaling",
            ),
        ]
        for config, message in cases:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                self.worker.validate_native_model_context(
                    config, {"max_position_embeddings": 4096}
                )

    def test_native_context_preflight_exits_two_before_tokenizer_or_weights(self):
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
            with mock.patch.object(
                self.worker, "_load_native_config", create=True,
                return_value=types.SimpleNamespace(
                    max_position_embeddings=32, rope_scaling=None
                ),
            ) as preflight, mock.patch.object(self.worker, "_prepare_inputs") as prepare, \
                 mock.patch.object(self.worker, "_load_model") as load:
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 2)
            preflight.assert_called_once()
            prepare.assert_not_called()
            load.assert_not_called()
            failures = list((root / "out" / "failures").glob("*.json"))
            self.assertEqual(len(failures), 1)
            failure = json.loads(failures[0].read_text(encoding="utf-8"))
            self.assertEqual((failure["retryable"], failure["stage"]), (False, "input"))

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
        for record in records:
            self.assertEqual(record["topk_attention_overlap"], 1.0)
            self.assertTrue(all(
                record[name] == 0.0
                for name in schema
                if name != "topk_attention_overlap"
            ))

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
            with mock.patch.object(self.worker, "point_id", return_value="completed"), \
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
            run_id = cage_io.point_id(job, sample, 2, source)
            stale_run, stale_layers = completed_artifacts(
                run_id, layer_count=1, prompt_length=2
            )
            stale_run["metrics_aggregate"][
                "relative_k_reconstruction_error"
            ]["mean"] = 1.0
            atomic_write_json(root / "out" / "runs" / f"{run_id}.json", stale_run)
            atomic_write_jsonl(
                root / "out" / "layers" / f"{run_id}.jsonl", stale_layers
            )
            prepared = {
                ("s", 2): {
                    "prompt_ids": self_tensor.tensor([[1, 2]]),
                    "continuation_ids": self_tensor.tensor([[3]]),
                    "text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                }
            }
            with mock.patch.object(self.worker, "collect_provenance", return_value=provenance), \
                 mock.patch.object(
                     self.worker, "_load_native_config",
                     return_value=types.SimpleNamespace(
                         model_type="llama",
                         max_position_embeddings=16,
                         rope_scaling=None,
                     ),
                 ), \
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

    def test_cuda_peak_reset_precedes_reference_pass(self):
        events = []
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cuda",
                      "max_position_embeddings": 16},
            "prompts_file": "prompts.jsonl", "sample_ids": ["s"],
            "prompt_lengths": [2],
            "methods": [{"id": "fp", "method": "fp16", "method_config": {}}],
            "measurement": {"decode_tokens": 1, "seed": 7}, "output_dir": "out",
        }

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(model_type="llama")
                self.model = types.SimpleNamespace(layers=[object()])

            def __call__(self, *, input_ids, use_cache, return_dict, past_key_values=None):
                del return_dict
                if not use_cache:
                    events.append("reference")
                elif past_key_values is None:
                    events.append("prefill")
                else:
                    events.append("decode")
                output = types.SimpleNamespace(logits=self_tensor.zeros(1, 1, 2))
                if use_cache:
                    if past_key_values is None:
                        key = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                        value = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                        output.past_key_values = ((key, value),)
                    else:
                        output.past_key_values = past_key_values
                return output

        self_tensor = self.worker.torch
        source = {"git_commit": "abc", "dirty": False, "dirty_sha256": None}
        provenance = {"source_state": source, "deterministic_seed": 7}
        prepared = {
            ("s", 2): {
                "prompt_ids": self_tensor.tensor([[1, 2]]),
                "continuation_ids": self_tensor.tensor([[3]]),
                "text_sha256": "0" * 64,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prompts.jsonl").write_text(
                json.dumps({"sample_id": "s", "text": "hello"}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                self.worker, "collect_provenance", return_value=provenance
            ), mock.patch.object(
                self.worker, "_load_native_config",
                return_value=types.SimpleNamespace(
                    model_type="llama", max_position_embeddings=16, rope_scaling=None,
                ),
            ), mock.patch.object(
                self.worker, "_prepare_inputs", return_value=(object(), prepared)
            ), mock.patch.object(
                self.worker, "_load_model", return_value=(FakeModel(), 0.1)
            ), mock.patch.object(
                self.worker.torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor
            ), mock.patch.object(
                self.worker.torch.cuda, "reset_peak_memory_stats",
                side_effect=lambda: events.append("reset_peak"),
            ), mock.patch.object(
                self.worker.torch.cuda, "max_memory_allocated", return_value=0
            ), mock.patch.object(
                self.worker.torch.cuda, "max_memory_reserved", return_value=0
            ), mock.patch.object(self.worker.torch.cuda, "empty_cache"):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 0)

        self.assertEqual(events[:2], ["reset_peak", "reference"])

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
             ("experiment_point_error", 6, False)),
            (RuntimeError("decode failed"), "decode", False,
             ("experiment_point_error", 6, False)),
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

    def test_isolated_point_failure_returns_six_and_continues_later_points(self):
        manifest = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "prompts_file": "prompts.jsonl", "sample_ids": ["s"],
            "prompt_lengths": [2, 3],
            "methods": [{"id": "fp", "method": "fp16", "method_config": {}}],
            "measurement": {"decode_tokens": 1, "seed": 7}, "output_dir": "out",
        }

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(model_type="llama")
                self.model = types.SimpleNamespace(layers=[object()])
                self.failed_once = False

            def __call__(self, *, input_ids, use_cache, return_dict, past_key_values=None):
                del return_dict
                if not use_cache and input_ids.shape[-1] == 3 and not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("isolated deterministic point failure")
                output = types.SimpleNamespace(logits=self_tensor.zeros(1, 1, 2))
                if use_cache and past_key_values is None:
                    key = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                    value = self_tensor.zeros(1, 1, input_ids.shape[-1], 2)
                    output.past_key_values = ((key, value),)
                return output

        self_tensor = self.worker.torch
        source = {"git_commit": "abc", "dirty": False, "dirty_sha256": None}
        provenance = {"source_state": source, "deterministic_seed": 7}
        prepared = {
            ("s", length): {
                "prompt_ids": self_tensor.arange(length).unsqueeze(0),
                "continuation_ids": self_tensor.tensor([[9]]),
                "text_sha256": "0" * 64,
            }
            for length in (2, 3)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prompts.jsonl").write_text(
                json.dumps({"sample_id": "s", "text": "hello"}) + "\n", encoding="utf-8"
            )
            with mock.patch.object(self.worker, "collect_provenance", return_value=provenance), \
                 mock.patch.object(self.worker, "_load_native_config", create=True,
                                   return_value=types.SimpleNamespace(
                                       model_type="llama",
                                       max_position_embeddings=16,
                                       rope_scaling=None,
                                   )), \
                 mock.patch.object(self.worker, "_prepare_inputs",
                                   return_value=(object(), prepared)), \
                 mock.patch.object(self.worker, "_load_model", return_value=(FakeModel(), 0.1)):
                self.assertEqual(self.worker.run_job(root / "manifest.json", 0), 6)
            failures = list((root / "out" / "failures").glob("*.json"))
            runs = list((root / "out" / "runs").glob("*.json"))
            self.assertEqual((len(failures), len(runs)), (1, 1))
            failure = json.loads(failures[0].read_text(encoding="utf-8"))
            self.assertEqual(
                (failure["category"], failure["retryable"]),
                ("experiment_point_error", False),
            )


if __name__ == "__main__":
    unittest.main()
