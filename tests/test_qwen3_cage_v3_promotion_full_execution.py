import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_full import (
    CageV3PromotionFullError,
    EXPECTED_CASE_ID_HASHES,
    load_full_execution,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_full_execution_v1.json"


class CageV3PromotionFullExecutionTest(unittest.TestCase):
    def test_checked_in_execution_freezes_exact_grid_resume_and_boundaries(self):
        execution, execution_sha256, protocol, protocol_sha256, gate, gate_sha256 = load_full_execution(
            EXECUTION_PATH,
            repo_root=REPO_ROOT,
            verify_server_artifacts=False,
        )
        self.assertEqual(len(execution_sha256), 64)
        self.assertEqual(len(protocol_sha256), 64)
        self.assertEqual(len(gate_sha256), 64)
        self.assertEqual(execution["partitions"]["cage_qwen3"]["case_count"], 480)
        self.assertEqual(execution["partitions"]["kitty_qwen3"]["case_count"], 120)
        self.assertEqual(execution["partitions"]["cage_qwen3"]["case_ids_sha256"], EXPECTED_CASE_ID_HASHES["cage_qwen3"])
        self.assertEqual(execution["partitions"]["kitty_qwen3"]["case_ids_sha256"], EXPECTED_CASE_ID_HASHES["kitty_qwen3"])
        self.assertEqual(execution["execution_order"], ["kitty_qwen3", "cage_qwen3"])
        self.assertTrue(execution["authorization"]["gpu_full_holdout_execution"])
        self.assertFalse(execution["authorization"]["holdout_interpretation"])
        self.assertFalse(execution["authorization"]["pg19_test"])

    def test_execution_rejects_case_scope_resume_or_authorization_mutation(self):
        execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(execution)
        changed["partitions"]["kitty_qwen3"]["case_count"] = 121
        mutations.append(changed)
        changed = copy.deepcopy(execution)
        changed["resume_policy"]["cross_partition_output_reuse"] = True
        mutations.append(changed)
        changed = copy.deepcopy(execution)
        changed["authorization"]["holdout_interpretation"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "execution.json"
                    path.write_text(json.dumps(mutation), encoding="utf-8")
                    with self.assertRaises(CageV3PromotionFullError):
                        load_full_execution(
                            path,
                            repo_root=REPO_ROOT,
                            verify_server_artifacts=False,
                        )

    def test_runner_exposes_only_partition_execution_and_resume(self):
        source = (REPO_ROOT / "scripts" / "qwen3_run_cage_v3_promotion_full.py").read_text(encoding="utf-8")
        self.assertIn('choices=PARTITIONS', source)
        self.assertIn('resume-valid', source)
        self.assertIn('"holdout_interpretation_authorized": False', source)
        self.assertIn('"pg19_test_accessed": False', source)
        self.assertNotIn("bootstrap", source)
        self.assertNotIn("relative_ppl_percent", source)
        self.assertNotIn("promotion_gate_pass", source)


if __name__ == "__main__":
    unittest.main()
