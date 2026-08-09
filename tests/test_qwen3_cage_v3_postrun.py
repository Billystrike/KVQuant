import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.qwen3_cage_v3_postrun import (
    CageV3PostrunError,
    _validate_cache,
    shell_case_manifest_sha256,
    validate_artifact_manifest,
    validate_failed_pre_case_attempt,
    validate_failed_validation_attempt,
    validate_validation_attempt_manifest,
)
from utils.qwen3_cage_v3_protocol import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_artifacts_v1.json"
EXPECTED_ARTIFACT_SHA256 = "0543fa82f53127ce242fc0c89beeaa3867384a91826c75ff88122e6e91b882cf"
ATTEMPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_postrun_attempts_v1.json"
EXPECTED_ATTEMPT_SHA256 = "e13629915023b4e570c77b60f319fa88bc49181d94f34e016556efb32b75893d"


class CageV3PostrunTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.attempts = json.loads(ATTEMPT_PATH.read_text(encoding="utf-8"))

    def _validate(self, value):
        validate_artifact_manifest(
            value,
            execution_sha256="2f94fdb35c1c087270b971b524cb9a26722676d1d950fa8588037027a90f7dc0",
            protocol_sha256="c92e452b5eac99c7a015da21a82080cf61ddd070e677cc3664ed78bf657cbfe9",
            input_manifest_sha256="e53cdec987d8206ed6c21ba3be5c04f3751c25f2fee074fa12e30916e06404d7",
            quota_plan_sha256="01a1063651d4565a732474b9f27f7a674f434750e7aea2f55429f619bed91914",
            gate_sha256="89329379bed30bb82b7974252bc453156b27f503d39331a80de4a0d523b09d7f",
        )

    def test_checked_in_artifacts_freeze_75_cases_and_failed_pre_case_attempt(self):
        self.assertEqual(file_sha256(ARTIFACT_PATH), EXPECTED_ARTIFACT_SHA256)
        self._validate(self.artifacts)
        self.assertEqual(self.artifacts["joint_expected"]["case_count"], 75)
        self.assertEqual(self.artifacts["joint_expected"]["layer_record_count"], 2700)
        self.assertEqual(self.artifacts["joint_expected"]["failed_pre_case_attempt_count"], 1)
        self.assertFalse(self.artifacts["claim_eligible"])
        self.assertFalse(self.artifacts["interpretation_performed"])

    def test_artifact_manifest_rejects_scientific_and_provenance_mutation(self):
        mutations = []
        value = copy.deepcopy(self.artifacts)
        value["partitions"]["cage_qwen3"]["case_count"] = 59
        mutations.append((value, "case count"))
        value = copy.deepcopy(self.artifacts)
        value["joint_expected"]["layer_record_count"] = 2699
        mutations.append((value, "expectations"))
        value = copy.deepcopy(self.artifacts)
        value["failed_pre_case_attempts"][0]["case_execution_started"] = True
        mutations.append((value, "executed a case"))
        value = copy.deepcopy(self.artifacts)
        value["interpretation_performed"] = True
        mutations.append((value, "interpretation"))
        for payload, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CageV3PostrunError, message):
                    self._validate(payload)

    def test_cache_audit_accepts_per_layer_two_bit_quotas_and_uniform_value(self):
        quotas = [48] * 12 + [32] * 12 + [16] * 12
        record = {
            "method": {
                "name": "cage_v2",
                "family_id": "pure-sr2-sink32-calibrated",
                "config": {
                    "residual_length": 128,
                    "sink_length": 32,
                    "one_bit_channels": 0,
                    "two_bit_channels": quotas,
                },
            },
            "input": {"prompt_length": 1024},
            "cache": {
                "reported_seq_length": 1025,
                "expected_seq_length": 1025,
                "layer_count": 36,
                "tensor_dtypes": ["torch.float16"],
                "tensors_finite": True,
                "key_quantized_lengths": [928],
                "value_quantized_lengths": [897],
                "value_adaptive": False,
                "one_bit_channels": 0,
                "two_bit_channels": quotas,
            },
        }
        _validate_cache(record)
        record["cache"]["value_adaptive"] = True
        with self.assertRaisesRegex(CageV3PostrunError, "Value adaptation"):
            _validate_cache(record)

    def test_cache_audit_distinguishes_v2_controls_from_v3_candidates(self):
        control = {
            "method": {
                "name": "cage_v2",
                "family_id": "cage-v2-best-control",
                "config": {
                    "residual_length": 128,
                    "sink_length": 32,
                    "one_bit_channels": 16,
                    "two_bit_channels": 16,
                },
            },
            "input": {"prompt_length": 1024},
            "cache": {
                "reported_seq_length": 1025,
                "expected_seq_length": 1025,
                "layer_count": 36,
                "tensor_dtypes": ["torch.float16"],
                "tensors_finite": True,
                "key_quantized_lengths": [928],
                "value_quantized_lengths": [897],
                "value_adaptive": False,
                "one_bit_channels": 16,
                "two_bit_channels": 16,
            },
        }
        _validate_cache(control)
        candidate = copy.deepcopy(control)
        candidate["method"]["family_id"] = "pure-sr2-sink32-uniform"
        with self.assertRaisesRegex(CageV3PostrunError, "one-bit refinement"):
            _validate_cache(candidate)

    def test_failed_postrun_validation_is_frozen_without_scientific_mutation(self):
        self.assertEqual(file_sha256(ATTEMPT_PATH), EXPECTED_ATTEMPT_SHA256)
        validate_validation_attempt_manifest(self.attempts)
        self.assertEqual(len(self.attempts["attempts"]), 5)
        attempt = self.attempts["attempts"][0]
        self.assertFalse(attempt["output_created"])
        self.assertFalse(attempt["scientific_artifacts_mutated"])
        self.assertFalse(attempt["interpretation_performed"])

    def test_validation_attempt_manifest_rejects_mutation(self):
        for field, value, message in (
            ("scientific_artifacts_mutated", True, "mutated"),
            ("interpretation_performed", True, "interpretation"),
        ):
            payload = copy.deepcopy(self.attempts)
            payload["attempts"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(CageV3PostrunError, message):
                    validate_validation_attempt_manifest(payload)

    def test_shell_manifest_uses_cases_relative_paths(self):
        with TemporaryDirectory() as temporary:
            cases = Path(temporary) / "cases"
            cases.mkdir()
            paths = [cases / "b.json", cases / "a.json"]
            for path in paths:
                path.write_text(path.stem, encoding="utf-8")
            lines = "".join(
                f"{file_sha256(path)}  cases/{path.name}\n"
                for path in sorted(paths)
            )
            expected = hashlib.sha256(lines.encode("utf-8")).hexdigest()
            self.assertEqual(shell_case_manifest_sha256(paths), expected)

    def test_failed_validation_log_uses_attempt_specific_test_count(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "attempt.log"
            output_path = root / "absent.json"
            message = "synthetic validator failure"
            log_text = (
                "Ran 30 tests in 1.000s\n\nOK\n"
                "=== JOINT 75-CASE DEEP AUDIT ===\n"
                f"CageV3PostrunError: {message}\n"
            )
            log_path.write_text(log_text, encoding="utf-8")
            attempt = {
                "attempt_index": 2,
                "source_commit": "1" * 40,
                "validator_sha256": "2" * 64,
                "postrun_utils_sha256": "3" * 64,
                "execution_log": str(log_path),
                "execution_log_sha256": file_sha256(log_path),
                "execution_log_size_bytes": log_path.stat().st_size,
                "intended_output": str(output_path),
                "output_created": False,
                "tests_passed_before_failure": 30,
                "failure_type": "CageV3PostrunError",
                "failure_message": message,
                "scientific_artifacts_mutated": False,
                "interpretation_performed": False,
            }
            report = validate_failed_validation_attempt(attempt)
            self.assertEqual(report["attempt_index"], 2)
            attempt["tests_passed_before_failure"] = 28
            with self.assertRaisesRegex(CageV3PostrunError, "test evidence"):
                validate_failed_validation_attempt(attempt)

    def test_pre_case_failure_log_does_not_require_outer_pipeline_status(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "pre_case.log"
            script_path = root / "failed.sh"
            observed = "dfd2c07b407179359207c612ab631f3ed1"
            expected = "dfd2c07b407d6b407179359207c612ab631f3ed1"
            log_path.write_text(
                "=== QWEN3 CAGE-V3 SCREEN KITTY FULL START ===\n"
                "ERROR: unexpected Kitty HEAD\n",
                encoding="utf-8",
            )
            script_path.write_text(f"EXPECTED_KITTY={observed}\n", encoding="utf-8")
            attempt = {
                "partition": "kitty_qwen3",
                "reason": "truncated_noncanonical_expected_kitty_commit_literal",
                "case_execution_started": False,
                "execution_log": str(log_path),
                "execution_log_sha256": file_sha256(log_path),
                "execution_log_size_bytes": log_path.stat().st_size,
                "archived_script": str(script_path),
                "archived_script_sha256": file_sha256(script_path),
                "observed_literal": observed,
                "observed_literal_length": len(observed),
                "observed_literal_sha256": hashlib.sha256(observed.encode("ascii")).hexdigest(),
                "expected_literal": expected,
                "expected_literal_length": len(expected),
                "expected_literal_sha256": hashlib.sha256(expected.encode("ascii")).hexdigest(),
            }
            report = validate_failed_pre_case_attempt(attempt)
            self.assertFalse(report["case_execution_started"])


if __name__ == "__main__":
    unittest.main()
