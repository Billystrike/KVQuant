import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_cage_experiment_schema import completed_artifacts
import utils.cage_experiment_io as cage_io

from utils.cage_experiment_io import (
    aggregate_completed_runs,
    atomic_write_json,
    atomic_write_jsonl,
    collect_provenance,
    load_prompt_records,
    normalize_command_arguments,
    prepare_prompt,
    stable_run_id,
    source_state_identity,
)


class _Tokenizer:
    def __call__(self, text, **kwargs):
        self.call = (text, kwargs)
        return {"input_ids": torch.tensor([[9, 8, 7, 6]])}


class CageExperimentIOTests(unittest.TestCase):
    @staticmethod
    def _completed_record(run_id, **updates):
        record, _ = completed_artifacts(run_id)
        record.update(updates)
        return record

    @staticmethod
    def _write_completed(root, run_id, *, runtime_updates=None):
        record, layers = completed_artifacts(run_id)
        record["runtime_diagnostics"].update(runtime_updates or {})
        atomic_write_json(root / "runs" / f"{run_id}.json", record)
        atomic_write_jsonl(root / "layers" / f"{run_id}.jsonl", layers)

    def test_prompt_jsonl_is_strict_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('{"sample_id":"a","text":"hello"}\n', encoding="utf-8")
            records = load_prompt_records(path)
            self.assertEqual(records["a"].text, "hello")
            path.write_text('{"sample_id":"a","text":"x"}\n{"sample_id":"a","text":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_prompt_records(path)
            path.write_text('{"text":"x"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample_id"):
                load_prompt_records(path)

    def test_prompt_jsonl_rejects_representative_malformed_records(self):
        cases = {
            "invalid JSON": "{not-json}\n",
            "JSON object": "[]\n",
            "unknown fields": '{"sample_id":"a","text":"x","extra":1}\n',
            "sample_id": '{"sample_id":"","text":"x"}\n',
            "text": '{"sample_id":"a","text":"  "}\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            for message, contents in cases.items():
                with self.subTest(message=message):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_prompt_records(path)

    def test_prompt_jsonl_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one record"):
                load_prompt_records(path)

    def test_prepare_prompt_uses_exact_t_and_t_plus_one(self):
        tokenizer = _Tokenizer()
        result = prepare_prompt(tokenizer, "hello", 3)
        self.assertEqual(result["prompt_ids"].tolist(), [[9, 8, 7]])
        self.assertEqual(result["continuation_ids"].tolist(), [[6]])
        self.assertTrue(result["prompt_ids"].is_contiguous())
        self.assertEqual(tokenizer.call[1], {"add_special_tokens": True, "return_tensors": "pt"})
        with self.assertRaisesRegex(ValueError, "need 5"):
            prepare_prompt(tokenizer, "hello", 4)

    def test_run_id_is_canonical(self):
        self.assertEqual(stable_run_id({"b": 2, "a": {"d": 4, "c": 3}}),
                         stable_run_id({"a": {"c": 3, "d": 4}, "b": 2}))

    def test_point_id_is_stable_across_acceptance_and_full_matrix_scope(self):
        sample = cage_io.PromptRecord("sample", "same scientific input")
        source = {"git_commit": "abc", "dirty": False, "dirty_sha256": None}
        acceptance = {
            "job_id": "acceptance",
            "model": {"reference": r"C:\models\llama", "dtype": "float16",
                      "device": "cuda", "max_position_embeddings": 4096},
            "method": "cage",
            "method_config": {"k_bits": 2, "v_bits": 2, "residual_length": 32},
            "measurement": {"decode_tokens": 1, "seed": 7},
            "sample_ids": ["sample"],
            "prompt_lengths": [512],
            "prompts_file": r"C:\acceptance\prompts.jsonl",
            "output_dir": r"C:\results",
        }
        full = json.loads(json.dumps(acceptance))
        full.update({
            "job_id": "full",
            "sample_ids": ["sample", "other"],
            "prompt_lengths": [512, 1024],
            "prompts_file": r"D:\full\prompts.jsonl",
            "output_dir": r"D:\elsewhere",
        })

        self.assertEqual(
            cage_io.point_id(acceptance, sample, 512, source),
            cage_io.point_id(full, sample, 512, source),
        )

    def test_provenance_has_cpu_cuda_fields(self):
        provenance = collect_provenance(Path(__file__).parents[1], deterministic_seed=123)
        for key in ("source_state", "dirty", "python", "pytorch", "transformers",
                    "cuda_runtime", "cuda_driver", "gpu_name", "deterministic_seed", "command"):
            self.assertIn(key, provenance)
        if not torch.cuda.is_available():
            self.assertIsNone(provenance["cuda_runtime"])
            self.assertIsNone(provenance["cuda_driver"])
            self.assertIsNone(provenance["gpu_name"])
        self.assertEqual(provenance["deterministic_seed"], 123)

    def test_provenance_captures_determinism_settings_without_toggling_them(self):
        with mock.patch.object(
            torch, "are_deterministic_algorithms_enabled", return_value=True
        ), mock.patch.object(
            torch, "is_deterministic_algorithms_warn_only_enabled",
            create=True, return_value=True,
        ), mock.patch.object(
            torch, "use_deterministic_algorithms"
        ) as toggle, mock.patch.dict(
            os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        ):
            provenance = collect_provenance(
                Path(__file__).parents[1], deterministic_seed=321
            )
        self.assertEqual(provenance["deterministic_seed"], 321)
        self.assertEqual({
            "deterministic_algorithms_enabled": provenance.get(
                "deterministic_algorithms_enabled"
            ),
            "deterministic_algorithms_warn_only": provenance.get(
                "deterministic_algorithms_warn_only"
            ),
            "cudnn_deterministic": provenance.get("cudnn_deterministic"),
            "cudnn_benchmark": provenance.get("cudnn_benchmark"),
            "cublas_workspace_config": provenance.get("cublas_workspace_config"),
        }, {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": True,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": ":4096:8",
        })
        toggle.assert_not_called()

    def test_command_path_arguments_are_normalized_relative_to_repo_root(self):
        root = Path(__file__).parents[1].resolve()
        expected_manifest = (root / "config" / "manifest.json").resolve().as_posix()
        expected_output = (root / "results").resolve().as_posix()
        expected_prompts = (root / "data" / "prompts.jsonl").resolve().as_posix()
        first = normalize_command_arguments([
            "runner.py", "--manifest", ".\\config\\.\\manifest.json",
            "--output-dir=.\\tmp\\..\\results", "--prompts-file", "data/./prompts.jsonl",
            "--job-index", "3",
        ], root)
        second = normalize_command_arguments([
            "runner.py", "--manifest", "config/manifest.json",
            "--output-dir=results", "--prompts-file", "data/prompts.jsonl",
            "--job-index", "3",
        ], root)
        self.assertEqual(first, second)
        self.assertEqual(first, [
            "runner.py", "--manifest", expected_manifest,
            f"--output-dir={expected_output}", "--prompts-file", expected_prompts,
            "--job-index", "3",
        ])
        self.assertEqual(
            normalize_command_arguments(["runner.py", "--manifest=config/../manifest.json"], root),
            ["runner.py", f"--manifest={(root / 'manifest.json').as_posix()}"],
        )

    def test_provenance_stores_normalized_command_arguments(self):
        root = Path(__file__).parents[1].resolve()
        argv = ["runner.py", "--manifest", ".\\config\\..\\manifest.json", "--job-index=2"]
        with mock.patch("utils.cage_experiment_io.sys.argv", argv):
            provenance = collect_provenance(root, deterministic_seed=17)
        self.assertEqual(provenance["command"], [
            "runner.py", "--manifest", (root / "manifest.json").as_posix(), "--job-index=2",
        ])
        self.assertEqual(provenance["deterministic_seed"], 17)

    def test_source_identity_hashes_untracked_code_but_not_output_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.txt").write_text("base", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            (root / "result.json").write_text("{}", encoding="utf-8")
            self.assertFalse(source_state_identity(root)["dirty"])
            (root / "experiment.py").write_text("VALUE = 1\n", encoding="utf-8")
            first = source_state_identity(root)
            self.assertTrue(first["dirty"])
            self.assertEqual(first["untracked_paths"], ["experiment.py"])
            (root / "experiment.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(first["dirty_sha256"], source_state_identity(root)["dirty_sha256"])

    def test_atomic_writers_replace_and_leave_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            path.write_text("old", encoding="utf-8")
            atomic_write_json(path, {"new": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": 1})
            lines = Path(directory) / "value.jsonl"
            atomic_write_jsonl(lines, [{"z": 1, "a": "snow"}, {"b": 2, "a": 1}])
            self.assertEqual(lines.read_bytes(), b'{"a":"snow","z":1}\n{"a":1,"b":2}\n')
            self.assertEqual([json.loads(x) for x in lines.read_text(encoding="utf-8").splitlines()],
                             [{"a": "snow", "z": 1}, {"a": 1, "b": 2}])
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["value.json", "value.jsonl"])

    def test_aggregation_only_includes_completed_and_flattens_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir(); (root / "failures").mkdir()
            atomic_write_json(root / "runs" / "b.json", {"run_id": "b", "status": "failed"})
            self._write_completed(root, "a", runtime_updates={"score": 1})
            atomic_write_json(root / "failures" / "c.json", {"run_id": "c", "status": "failed"})
            records = aggregate_completed_runs(root)
            self.assertEqual([x["run_id"] for x in records], ["a"])
            with (root / "summary" / "runs.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["model.reference"], "model")
            self.assertEqual(row["runtime_diagnostics.score"], "1")

    def test_aggregation_excludes_valid_old_source_and_removed_config_points(self):
        sample = cage_io.PromptRecord("sample", "selected prompt")
        source = {"git_commit": "current", "dirty": False, "dirty_sha256": None}
        job = {
            "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                      "max_position_embeddings": 16},
            "method": "fp16",
            "method_config": {},
            "measurement": {"decode_tokens": 1, "seed": 7},
        }
        current_id = cage_io.point_id(job, sample, 2, source)
        old_source_id = cage_io.point_id(
            job, sample, 2,
            {"git_commit": "old", "dirty": False, "dirty_sha256": None},
        )
        removed_job = json.loads(json.dumps(job))
        removed_job.update({
            "method": "kivi",
            "method_config": {
                "k_bits": 2, "v_bits": 2, "group_size": 2, "residual_length": 2,
            },
        })
        removed_config_id = cage_io.point_id(removed_job, sample, 2, source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = (
                (current_id, job, source),
                (
                    old_source_id,
                    job,
                    {"git_commit": "old", "dirty": False, "dirty_sha256": None},
                ),
                (removed_config_id, removed_job, source),
            )
            for run_id, artifact_job, artifact_source in artifacts:
                record, layers = completed_artifacts(run_id, prompt_length=2)
                record["method"] = {
                    "name": artifact_job["method"],
                    "resolved_config": artifact_job["method_config"],
                }
                record["quantization"] = {
                    "method": artifact_job["method"], **artifact_job["method_config"],
                }
                record["provenance"]["source_state"] = artifact_source
                for layer in layers:
                    layer["method"] = artifact_job["method"]
                atomic_write_json(root / "runs" / f"{run_id}.json", record)
                atomic_write_jsonl(root / "layers" / f"{run_id}.jsonl", layers)
            records = aggregate_completed_runs(root, expected_run_ids={current_id})

        self.assertEqual([record["run_id"] for record in records], [current_id])

    def test_aggregation_rejects_invalid_expected_point_but_ignores_unrelated_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_completed(root, "expected")
            self._write_completed(root, "historical")
            (root / "layers" / "expected.jsonl").write_text(
                "{bad json}\n", encoding="utf-8"
            )
            (root / "runs" / "historical.json").write_text(
                "{bad json}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "layer.*JSON"):
                aggregate_completed_runs(root, expected_run_ids={"expected"})

    def test_aggregation_csv_uses_union_of_completed_record_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir()
            self._write_completed(root, "a", runtime_updates={"left": 1})
            self._write_completed(root, "b", runtime_updates={"right": 2})
            aggregate_completed_runs(root)
            with (root / "summary" / "runs.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((rows[0]["run_id"], rows[0]["runtime_diagnostics.left"], rows[0]["runtime_diagnostics.right"]),
                             ("a", "1", ""))
            self.assertEqual((rows[1]["run_id"], rows[1]["runtime_diagnostics.left"], rows[1]["runtime_diagnostics.right"]),
                             ("b", "", "2"))

    def test_aggregation_rejects_incompatible_completed_schema_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir()
            for version in (2, True):
                with self.subTest(schema_version=version):
                    atomic_write_json(root / "runs" / "a.json",
                                      self._completed_record("a", schema_version=version))
                    with self.assertRaisesRegex(ValueError, r"a\.json.*schema_version.*1"):
                        aggregate_completed_runs(root)

    def test_aggregation_rejects_completed_filename_run_id_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir()
            atomic_write_json(root / "runs" / "a.json", self._completed_record("b"))
            with self.assertRaisesRegex(ValueError, r"a\.json.*run_id.*filename"):
                aggregate_completed_runs(root)

    def test_aggregation_rejects_completed_record_missing_required_field_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir(); (root / "summary").mkdir()
            (root / "summary" / "runs.jsonl").write_text("existing-jsonl\n", encoding="utf-8")
            (root / "summary" / "runs.csv").write_text("existing-csv\n", encoding="utf-8")
            record = self._completed_record("a")
            del record["provenance"]
            atomic_write_json(root / "runs" / "a.json", record)
            with self.assertRaisesRegex(ValueError, r"a\.json.*missing required.*provenance"):
                aggregate_completed_runs(root)
            self.assertEqual((root / "summary" / "runs.jsonl").read_text(encoding="utf-8"),
                             "existing-jsonl\n")
            self.assertEqual((root / "summary" / "runs.csv").read_text(encoding="utf-8"),
                             "existing-csv\n")

    def test_aggregation_rejects_invalid_layer_artifact_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary").mkdir()
            (root / "summary" / "runs.jsonl").write_text("existing-jsonl\n", encoding="utf-8")
            (root / "summary" / "runs.csv").write_text("existing-csv\n", encoding="utf-8")
            self._write_completed(root, "a")
            (root / "layers" / "a.jsonl").write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layer.*JSON"):
                aggregate_completed_runs(root)
            self.assertEqual(
                (root / "summary" / "runs.jsonl").read_text(encoding="utf-8"),
                "existing-jsonl\n",
            )
            self.assertEqual(
                (root / "summary" / "runs.csv").read_text(encoding="utf-8"),
                "existing-csv\n",
            )

    def test_aggregation_rejects_corrupt_derived_aggregate_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary").mkdir()
            (root / "summary" / "runs.jsonl").write_text(
                "existing-jsonl\n", encoding="utf-8"
            )
            (root / "summary" / "runs.csv").write_text(
                "existing-csv\n", encoding="utf-8"
            )
            record, layers = completed_artifacts("a")
            record["metrics_aggregate"][
                "relative_k_reconstruction_error"
            ]["max"] = 1.0
            atomic_write_json(root / "runs" / "a.json", record)
            atomic_write_jsonl(root / "layers" / "a.jsonl", layers)
            with self.assertRaisesRegex(ValueError, "relative_k_reconstruction_error.max"):
                aggregate_completed_runs(root)
            self.assertEqual(
                (root / "summary" / "runs.jsonl").read_text(encoding="utf-8"),
                "existing-jsonl\n",
            )
            self.assertEqual(
                (root / "summary" / "runs.csv").read_text(encoding="utf-8"),
                "existing-csv\n",
            )

    def test_aggregation_with_no_completed_runs_writes_empty_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "runs").mkdir()
            atomic_write_json(root / "runs" / "failed.json", {"run_id": "x", "status": "failed"})
            self.assertEqual(aggregate_completed_runs(root), [])
            self.assertEqual((root / "summary" / "runs.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual((root / "summary" / "runs.csv").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
