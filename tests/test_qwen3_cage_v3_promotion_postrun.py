import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_postrun import (
    _validate_execution_log,
    scientific_payload,
    shell_case_manifest_sha256,
    validate_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_acceptance_artifacts_v1.json"


class CageV3PromotionPostrunTest(unittest.TestCase):
    def test_checked_in_artifact_manifest_freezes_acceptance_without_authorizing_full_holdout(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_artifact_manifest(manifest)
        self.assertEqual(manifest["partitions"]["cage_qwen3"]["case_count"], 12)
        self.assertEqual(manifest["partitions"]["kitty_qwen3"]["case_count"], 3)
        self.assertFalse(manifest["authorization_before_joint_gate"]["full_holdout"])

    def test_manifest_rejects_count_source_or_authorization_mutation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["partitions"]["kitty_qwen3"]["case_count"] = 4
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["acceptance_source"]["git_commit"] = "0" * 40
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["authorization_before_joint_gate"]["full_holdout"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                validate_artifact_manifest(mutation)

    def test_failed_attempt_receipts_must_remain_pre_case(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(manifest)
        changed["failed_pre_case_attempts"][0]["cases_executed"] = 1
        with self.assertRaisesRegex(RuntimeError, "executed cases"):
            validate_artifact_manifest(changed)

    def test_shell_case_manifest_uses_sorted_relative_case_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cases").mkdir()
            (root / "cases" / "b.json").write_text("b\n", encoding="utf-8")
            (root / "cases" / "a.json").write_text("a\n", encoding="utf-8")
            lines = "".join(
                f"{hashlib.sha256((root / 'cases' / name).read_bytes()).hexdigest()}  cases/{name}\n"
                for name in ("a.json", "b.json")
            )
            self.assertEqual(shell_case_manifest_sha256(root), hashlib.sha256(lines.encode()).hexdigest())

    def test_scientific_payload_excludes_runtime_and_is_order_deterministic(self):
        def record(case_id: str, runtime: float):
            return {
                "case_id": case_id,
                "partition": "cage_qwen3",
                "method": {"id": case_id},
                "input": {"prompt_length": 1024},
                "memory": {"model_total_bytes": 1},
                "scoring": {"token_nlls": [1.0]},
                "quality_cache": {"tensors_finite": True},
                "runtime": {"elapsed_seconds": runtime},
                "completed_at_utc": str(runtime),
            }
        first = scientific_payload([record("b", 1.0), record("a", 2.0)])
        second = scientific_payload([record("a", 99.0), record("b", 98.0)])
        self.assertEqual(first, second)
        self.assertEqual([row["case_id"] for row in first], ["a", "b"])

    def test_execution_log_checks_only_markers_inside_tee(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.log"
            path.write_text(
                "=== CAGE-V3 PROMOTION CAGE ACCEPTANCE A START ===\n"
                "=== CAGE-V3 PROMOTION CAGE ACCEPTANCE A END ===\n",
                encoding="utf-8",
            )
            spec = {
                "execution_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "execution_log_size_bytes": path.stat().st_size,
            }
            checks = _validate_execution_log(path, spec, partition="cage_qwen3", repeat="a")
            self.assertTrue(all(checks.values()))
            self.assertNotIn("pass_marker", checks)
            self.assertNotIn("terminal_safety", checks)


if __name__ == "__main__":
    unittest.main()
