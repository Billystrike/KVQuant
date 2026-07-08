import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "cage_smoke.py"
    spec = importlib.util.spec_from_file_location("cage_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CageSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = _load_smoke_module()

    def test_configure_modes_sets_original_kivi_and_cage_fake_fields(self):
        original = SimpleNamespace()
        cage = SimpleNamespace()

        self.smoke.configure_generation_mode(original, mode="kivi")
        self.smoke.configure_generation_mode(
            cage,
            mode="cage_fake",
            collect_metrics=True,
            dump_dir="metrics-out",
        )

        self.assertFalse(original.cage_enable)
        self.assertTrue(cage.cage_enable)
        self.assertEqual(cage.cage_mode, "fake")
        self.assertTrue(cage.cage_memory_summary)
        self.assertTrue(cage.cage_collect_metrics)
        self.assertEqual(cage.cage_dump_dir, "metrics-out")

    def test_count_new_tokens_uses_prompt_length(self):
        input_ids = torch.tensor([[5, 6, 7]])
        generated_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])

        self.assertEqual(self.smoke.count_new_tokens(generated_ids, input_ids), 3)

    def test_count_new_tokens_rejects_shorter_output_with_actionable_message(self):
        input_ids = torch.tensor([[5, 6, 7]])
        generated_ids = torch.tensor([[5, 6]])

        with self.assertRaisesRegex(ValueError, "shorter than the prompt"):
            self.smoke.count_new_tokens(generated_ids, input_ids)

    def test_validate_model_path_rejects_missing_local_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "Provide a valid --model"):
            self.smoke.validate_model_reference("not/a/local/model/path")

    def test_script_help_runs_when_executed_from_repo_root(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "cage_smoke.py"

        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=script_path.parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--model", completed.stdout)


if __name__ == "__main__":
    unittest.main()
