import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_cage_experiment_schema import completed_artifacts
from utils.cage_experiment_io import atomic_write_json, atomic_write_jsonl


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "cage_run_matrix.py"


def _load():
    spec = importlib.util.spec_from_file_location("cage_run_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(output_dir: Path, method_count: int = 2) -> dict:
    methods = [
        {"id": "fp16", "method": "fp16"},
        {
            "id": "kivi-g2-r2", "method": "kivi", "k_bits": 2, "v_bits": 2,
            "group_size": 2, "residual_length": 2,
        },
        {"id": "cage-r2", "method": "cage", "residual_length": 2},
    ]
    return {
        "model": {"reference": "model", "dtype": "float16", "device": "cpu",
                  "max_position_embeddings": 16},
        "prompts_file": str(output_dir.parent / "prompts.jsonl"),
        "sample_ids": ["sample"], "prompt_lengths": [2],
        "methods": methods[:method_count],
        "measurement": {"decode_tokens": 1, "seed": 7},
        "output_dir": str(output_dir),
    }


class CageRunMatrixTest(unittest.TestCase):
    def setUp(self):
        self.matrix = _load()

    def _write_manifest(self, root: Path, method_count: int = 2) -> Path:
        path = root / "manifest.json"
        path.write_text(json.dumps(_manifest(root / "output", method_count)), encoding="utf-8")
        return path

    def test_import_does_not_launch_workers(self):
        with mock.patch("subprocess.run") as run:
            _load()
        run.assert_not_called()

    def test_runs_one_worker_per_job_and_writes_resolved_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._write_manifest(root)
            with mock.patch.object(self.matrix.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 0)) as run:
                result = self.matrix.run_matrix(manifest_path)

            self.assertEqual(result, 0)
            self.assertEqual(run.call_count, 2)
            resolved_path = root / "output" / "manifest.resolved.json"
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [job["job_id"] for job in resolved["jobs"]],
                ["fp16", "kivi-g2-r2"],
            )
            for index, call in enumerate(run.call_args_list):
                command = call.args[0]
                self.assertEqual(command[-4:], ["--manifest", str(resolved_path),
                                                "--job-index", str(index)])
                self.assertEqual(call.kwargs, {"cwd": self.matrix.REPO_ROOT, "check": False})

    def test_deterministic_failure_is_not_retried_and_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_manifest(Path(directory))
            results = [subprocess.CompletedProcess([], 3), subprocess.CompletedProcess([], 0)]
            with mock.patch.object(self.matrix.subprocess, "run", side_effect=results) as run:
                result = self.matrix.run_matrix(manifest_path, retry_transient_once=True)
            self.assertEqual(result, 3)
            self.assertEqual(run.call_count, 2)

    def test_isolated_deterministic_exit_six_is_not_retried_and_later_jobs_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_manifest(Path(directory))
            results = [subprocess.CompletedProcess([], 6), subprocess.CompletedProcess([], 0)]
            with mock.patch.object(self.matrix.subprocess, "run", side_effect=results) as run:
                result = self.matrix.run_matrix(manifest_path, retry_transient_once=True)
            self.assertEqual(result, 6)
            self.assertEqual(run.call_count, 2)

    def test_transient_failure_is_retried_once_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_manifest(Path(directory), method_count=1)
            results = [subprocess.CompletedProcess([], 5), subprocess.CompletedProcess([], 0)]
            with mock.patch.object(self.matrix.subprocess, "run", side_effect=results) as run:
                self.assertEqual(self.matrix.run_matrix(manifest_path, True), 0)
            self.assertEqual(run.call_count, 2)

            with mock.patch.object(self.matrix.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 5)) as run:
                self.assertEqual(self.matrix.run_matrix(manifest_path, False), 5)
            self.assertEqual(run.call_count, 1)

    def test_aggregation_runs_after_failures_and_ignores_failed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._write_manifest(root, method_count=1)
            runs = root / "output" / "runs"
            failures = root / "output" / "failures"
            runs.mkdir(parents=True)
            failures.mkdir()
            run_record, layer_records = completed_artifacts("done")
            atomic_write_json(runs / "done.json", run_record)
            atomic_write_jsonl(root / "output" / "layers" / "done.jsonl", layer_records)
            (failures / "bad.json").write_text(
                json.dumps({"run_id": "bad", "status": "failed"}), encoding="utf-8")
            with mock.patch.object(self.matrix.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 4)):
                self.assertEqual(self.matrix.run_matrix(manifest_path), 4)
            lines = (root / "output" / "summary" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["run_id"] for line in lines], ["done"])
            csv_text = (root / "output" / "summary" / "runs.csv").read_text(encoding="utf-8")
            self.assertIn("done", csv_text)
            self.assertNotIn("bad", csv_text)

    def test_rejects_unknown_worker_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_manifest(Path(directory), method_count=1)
            with mock.patch.object(self.matrix.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 9)):
                with self.assertRaisesRegex(RuntimeError, "unexpected worker exit code 9"):
                    self.matrix.run_matrix(manifest_path)

    def test_resolved_manifest_preserves_relative_path_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(root / "unused", method_count=1)
            manifest["prompts_file"] = "inputs/prompts.jsonl"
            manifest["output_dir"] = "results"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(self.matrix.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 0)):
                self.matrix.run_matrix(manifest_path)
            resolved = json.loads((root / "results" / "manifest.resolved.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(resolved["prompts_file"], str((root / "inputs/prompts.jsonl").resolve()))
            self.assertEqual(resolved["jobs"][0]["prompts_file"], resolved["prompts_file"])
            self.assertEqual(resolved["jobs"][0]["output_dir"], resolved["output_dir"])

    def test_shared_input_failure_stops_later_jobs_and_still_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._write_manifest(root, method_count=3)
            with mock.patch.object(
                self.matrix.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 2),
            ) as run, mock.patch.object(self.matrix, "aggregate_completed_runs") as aggregate:
                result = self.matrix.run_matrix(manifest_path)
            self.assertEqual(result, 2)
            self.assertEqual(run.call_count, 1)
            aggregate.assert_called_once_with(root / "output")

    def test_transient_failure_twice_attempts_exactly_twice_then_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_manifest(Path(directory), method_count=2)
            results = [
                subprocess.CompletedProcess([], 5),
                subprocess.CompletedProcess([], 5),
                subprocess.CompletedProcess([], 0),
            ]
            with mock.patch.object(self.matrix.subprocess, "run", side_effect=results) as run:
                result = self.matrix.run_matrix(manifest_path, retry_transient_once=True)
            self.assertEqual(result, 5)
            self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
