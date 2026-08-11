import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_postrun import (
    CageV4PostrunError,
    shell_case_manifest_sha256,
    validate_artifact_manifest,
    validate_attempt_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_artifacts_v1.json"
ATTEMPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_postrun_attempts_v1.json"


class CageV4PostrunTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.attempts = json.loads(ATTEMPT_PATH.read_text(encoding="utf-8"))

    def _validate(self, value):
        validate_artifact_manifest(
            value,
            execution_sha256="3b64089d691f63b8cd544e9574eebf38b25252bd0d6199e17cc11a733849a573",
            protocol_sha256="ebbb974a2d5ef5c3ec249fef5c2e809f603db3937e3a5f799607422dba89dafe",
            input_manifest_sha256="7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d",
            gate_sha256="a443a84af5fdc01714852a6bde8145acb76295bf74e7327b5df9c7a9ba8988b8",
        )

    def test_checked_in_artifacts_freeze_exact_joint_counts_without_interpretation(self):
        self._validate(self.artifacts)
        self.assertEqual(self.artifacts["joint_expected"]["case_count"], 720)
        self.assertEqual(self.artifacts["joint_expected"]["target_token_count"], 46_080)
        self.assertEqual(self.artifacts["joint_expected"]["layer_record_count"], 21_600)
        self.assertFalse(self.artifacts["interpretation_performed"])
        self.assertFalse(self.artifacts["joint_expected"]["holdout_access"])

    def test_artifact_manifest_rejects_identity_count_or_boundary_mutation(self):
        mutations = (
            (("execution_sha256",), "0" * 64, "execution_sha256"),
            (("partitions", "cage_qwen3", "case_count"), 599, "case_count"),
            (("joint_expected", "holdout_access"), True, "expectations"),
            (("interpretation_performed",), True, "interpretation"),
        )
        for keys, replacement, message in mutations:
            value = copy.deepcopy(self.artifacts)
            target = value
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = replacement
            with self.subTest(keys=keys), self.assertRaisesRegex(CageV4PostrunError, message):
                self._validate(value)

    def test_shell_manifest_uses_relative_cases_paths_and_sorted_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "b.json"
            second = root / "a.json"
            first.write_text("b\n", encoding="utf-8")
            second.write_text("a\n", encoding="utf-8")
            observed = shell_case_manifest_sha256([first, second])
            import hashlib
            from utils.qwen3_cage_v4_data import file_sha256

            lines = (
                f"{file_sha256(second)}  cases/a.json\n"
                f"{file_sha256(first)}  cases/b.json\n"
            )
            self.assertEqual(observed, hashlib.sha256(lines.encode("utf-8")).hexdigest())

    def test_failed_marker_scope_attempt_is_frozen_without_interpretation(self):
        validate_attempt_manifest(self.attempts, verify_artifacts=False)
        attempt = self.attempts["attempts"][0]
        self.assertFalse(attempt["output_generated"])
        self.assertFalse(attempt["scientific_artifacts_mutated"])
        self.assertEqual(
            attempt["failure_type"],
            "audit_implementation_log_marker_scope_mismatch",
        )

    def test_attempt_manifest_rejects_hidden_or_reclassified_failure(self):
        for field, replacement in (
            ("failure_type", "scientific_failure"),
            ("scientific_artifacts_mutated", True),
            ("interpretation_performed", True),
        ):
            changed = copy.deepcopy(self.attempts)
            changed["attempts"][0][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                CageV4PostrunError, "attempt receipt"
            ):
                validate_attempt_manifest(changed, verify_artifacts=False)

    def test_execution_log_check_uses_in_tee_completion_markers(self):
        source = (REPO_ROOT / "utils" / "qwen3_cage_v4_postrun.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("CAGE-V4 METRIC SCREEN CAGE FULL END", source)
        self.assertIn("CAGE-V4 METRIC SCREEN KITTY FULL END", source)
        self.assertNotIn('result_marker = "CAGE_FULL_RESULT=PASS"', source)


if __name__ == "__main__":
    unittest.main()
