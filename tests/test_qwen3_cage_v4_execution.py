import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_execution import CageV4ExecutionError, load_screen_execution


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_execution_v1.json"


class CageV4ExecutionTest(unittest.TestCase):
    def test_execution_freezes_screen_counts_case_ids_and_receipts(self):
        execution, execution_sha256, protocol, protocol_sha256, manifest, expanded = (
            load_screen_execution(
                EXECUTION_PATH,
                repo_root=REPO_ROOT,
                verify_artifacts=False,
            )
        )
        self.assertEqual(len(execution_sha256), 64)
        self.assertEqual(protocol_sha256, execution["protocol"]["sha256"])
        self.assertIsNone(manifest)
        self.assertIsNone(expanded)
        self.assertEqual(execution["partitions"]["cage_qwen3"]["quality_case_count"], 600)
        self.assertEqual(execution["partitions"]["kitty_qwen3"]["quality_case_count"], 120)
        self.assertFalse(execution["execution_boundary"]["pg19_holdout_method_metrics"])
        self.assertFalse(execution["execution_boundary"]["pg19_test_access"])
        self.assertFalse(execution["execution_boundary"]["cage_v4_candidate_execution"])
        self.assertFalse(protocol["execution_boundary"]["holdout_method_metrics_authorized"])

    def test_execution_rejects_case_id_or_boundary_mutation(self):
        execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(execution)
        changed["partitions"]["cage_qwen3"]["case_ids_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(execution)
        changed["execution_boundary"]["pg19_test_access"] = True
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "execution.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(CageV4ExecutionError):
                        load_screen_execution(
                            path,
                            repo_root=REPO_ROOT,
                            verify_artifacts=False,
                        )

    def test_gpu_runner_has_no_stage_or_reserved_partition_switch(self):
        source = (
            REPO_ROOT / "scripts" / "qwen3_run_cage_v4_metric_screen.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("holdout", source.lower())
        self.assertNotIn("pg19_test", source.lower())
        self.assertIn("load_screen_execution", source)
        self.assertIn("verify_artifacts=True", source)
        self.assertIn("EXPECTED_KITTY_COMMIT", source)
        self.assertIn("EXPECTED_TRANSFORMERS_COMMIT", source)
        self.assertIn("runtime_dependency_state", source)
        validator = (
            REPO_ROOT / "scripts" / "qwen3_validate_cage_v4_metric_screen_execution.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_screen_execution", validator)
        self.assertIn("verify_artifacts=True", validator)


if __name__ == "__main__":
    unittest.main()
