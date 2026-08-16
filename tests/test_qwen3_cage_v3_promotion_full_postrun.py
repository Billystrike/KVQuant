import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_full_postrun import (
    _validate_execution_log,
    shell_case_manifest_sha256,
    validate_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_full_artifacts_v1.json"


class CageV3PromotionFullPostrunTest(unittest.TestCase):
    def test_checked_in_manifest_freezes_both_full_partitions_before_interpretation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_artifact_manifest(manifest)
        self.assertEqual(manifest["partitions"]["kitty_qwen3"]["case_count"], 120)
        self.assertEqual(manifest["partitions"]["cage_qwen3"]["case_count"], 480)
        self.assertFalse(manifest["postrun_boundary"]["analysis_authorized_by_manifest"])
        self.assertFalse(manifest["postrun_boundary"]["llama2_execution"])

    def test_manifest_rejects_result_identity_or_scope_mutation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["partitions"]["cage_qwen3"]["summary_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["failed_pre_case_attempts"][0]["cases_executed"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["postrun_boundary"]["analysis_authorized_by_manifest"] = True
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["source_sha256"]["runner"] = "0" * 64
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                validate_artifact_manifest(mutation)

    def test_shell_case_manifest_matches_relative_sorted_sha256sum_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cases").mkdir()
            (root / "cases" / "z.json").write_text("z\n", encoding="utf-8")
            (root / "cases" / "a.json").write_text("a\n", encoding="utf-8")
            lines = "".join(
                f"{hashlib.sha256((root / 'cases' / name).read_bytes()).hexdigest()}  cases/{name}\n"
                for name in ("a.json", "z.json")
            )
            self.assertEqual(shell_case_manifest_sha256(root), hashlib.sha256(lines.encode()).hexdigest())

    def test_execution_log_checks_only_inside_tee_start_and_end_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kitty.log"
            path.write_text(
                "=== CAGE-V3 PROMOTION KITTY FULL START ===\n"
                "=== CAGE-V3 PROMOTION KITTY FULL END ===\n",
                encoding="utf-8",
            )
            spec = {
                "execution_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "execution_log_size_bytes": path.stat().st_size,
            }
            checks = _validate_execution_log(path, spec, partition="kitty_qwen3")
            self.assertTrue(all(checks.values()))
            self.assertNotIn("result_marker", checks)
            self.assertNotIn("terminal_safety", checks)

    def test_postrun_source_does_not_interpret_metrics(self):
        source = (REPO_ROOT / "utils" / "qwen3_cage_v3_promotion_full_postrun.py").read_text(encoding="utf-8")
        self.assertNotIn("bootstrap_resamples", source)
        self.assertNotIn("relative_ppl_percent", source)
        self.assertNotIn("candidate_outcome", source)


if __name__ == "__main__":
    unittest.main()
