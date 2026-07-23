import json
import re
import tempfile
import unittest
from collections import UserDict
from pathlib import Path
from unittest import mock

import scripts.cage_run_passkey as passkey_runner
from utils.cage_passkey import (
    FILLER_ID,
    PASSKEY_SCHEMA_VERSION,
    PROMPT_TEMPLATE_ID,
    PasskeyError,
    aggregate_passkey_cases,
    build_passkey_prompt,
    expand_passkey_cases,
    first_five_digit,
    generate_passkeys,
    is_valid_completed_passkey_case,
    load_passkey_manifest,
    passkey_case_id,
    validate_completed_passkey_case,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self):
        self._vocabulary = {}

    def __call__(self, text, *, add_special_tokens):
        pieces = re.findall(r"\w+|[^\w\s]", text)
        ids = []
        if add_special_tokens:
            ids.append(1)
        for piece in pieces:
            if piece not in self._vocabulary:
                self._vocabulary[piece] = len(self._vocabulary) + 10
            ids.append(self._vocabulary[piece])
        return {"input_ids": ids}


class BatchEncodingLikeTokenizer(FakeTokenizer):
    """Exercise the mapping interface used by Transformers BatchEncoding."""

    def __call__(self, text, *, add_special_tokens):
        return UserDict(super().__call__(text, add_special_tokens=add_special_tokens))


def source_state():
    return {
        "git_commit": "abc",
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def completed_record(case, *, response_text=None):
    target = case["input"]["target"]
    response = response_text if response_text is not None else f" {target}"
    parsed = first_five_digit(response)
    generation = {
        "max_new_tokens": 8,
        "do_sample": False,
        "num_beams": 1,
        "seed": 0,
        "generated_token_count": 4,
        "response_text": response,
        "first_five_digit": parsed,
        "exact_match": parsed == target,
        "contains_target": target in response,
        "stopped_early": True,
    }
    model = {
        "reference": "/model",
        "dtype": "float16",
        "device": "cuda",
        "max_position_embeddings": 4096,
        "model_type": "llama",
    }
    method = {"id": "fp16", "name": "fp16", "resolved_config": {}}
    case_id = passkey_case_id(
        model=model,
        method=method,
        input_record=case["input"],
        generation=generation,
        source_state=source_state(),
    )
    return {
        "schema_version": PASSKEY_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "completed",
        "model": model,
        "method": method,
        "input": case["input"],
        "generation": generation,
        "runtime_diagnostics": {
            "elapsed_seconds": 1.25,
            "cuda_max_allocated_bytes": 100,
            "cuda_max_reserved_bytes": 120,
        },
        "provenance": {"source_state": source_state()},
    }


class PasskeyInputTests(unittest.TestCase):
    def test_generates_stable_unique_five_digit_keys(self):
        first = generate_passkeys(20260723, 5)
        second = generate_passkeys(20260723, 5)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(re.fullmatch(r"\d{5}", value) for value in first))

    def test_builds_exact_length_and_requested_statement_positions(self):
        tokenizer = FakeTokenizer()
        for position in (10, 50, 90):
            prepared = build_passkey_prompt(
                tokenizer,
                target="12345",
                prompt_length=512,
                position_percent=position,
            )
            self.assertEqual(len(prepared["input_ids"]), 512)
            self.assertAlmostEqual(
                prepared["actual_statement_position_fraction"],
                position / 100,
                delta=1 / 512,
            )
            self.assertLess(
                prepared["statement_token_start"], prepared["statement_token_end"]
            )

    def test_accepts_batch_encoding_like_mapping(self):
        prepared = build_passkey_prompt(
            BatchEncodingLikeTokenizer(),
            target="12345",
            prompt_length=512,
            position_percent=50,
        )
        self.assertEqual(len(prepared["input_ids"]), 512)

    def test_rejects_prompt_shorter_than_fixed_template(self):
        with self.assertRaisesRegex(PasskeyError, "shorter than fixed template"):
            build_passkey_prompt(
                FakeTokenizer(), target="12345", prompt_length=8, position_percent=50
            )

    def test_extracts_first_standalone_five_digit_value(self):
        self.assertEqual(first_five_digit("answer 12345 then 67890"), "12345")
        self.assertIsNone(first_five_digit("answer 1234 or 123456"))


class PasskeyManifestTests(unittest.TestCase):
    def test_checked_in_smoke_and_calibration_expand_and_reuse_smoke_cases(self):
        smoke = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json"
        )
        calibration = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_calibration.json"
        )
        _, smoke_cases = expand_passkey_cases(smoke, FakeTokenizer(), source_state())
        _, calibration_cases = expand_passkey_cases(
            calibration, FakeTokenizer(), source_state()
        )
        self.assertEqual(len(smoke_cases), 12)
        self.assertEqual(len(calibration_cases), 60)
        self.assertTrue(
            {case["case_id"] for case in smoke_cases}.issubset(
                {case["case_id"] for case in calibration_cases}
            )
        )
        self.assertEqual(smoke["output_dir"], calibration["output_dir"])

    def test_rejects_prompt_length_protocol_drift(self):
        valid = json.loads(
            (ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json").read_text()
        )
        valid["prompt_lengths"] = [4090]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(PasskeyError, "prompt_lengths must equal"):
                load_passkey_manifest(path)

    def test_rejects_non_fp16_stage_a_methods(self):
        valid = json.loads(
            (ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json").read_text()
        )
        valid["methods"].append({"id": "cage-r32", "method": "cage"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(PasskeyError, "Stage-A passkey methods"):
                load_passkey_manifest(path)

    def test_rejects_undeclared_stage_a_key_count(self):
        valid = json.loads(
            (ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json").read_text()
        )
        valid["key_generation"]["count"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(PasskeyError, "count must equal 1"):
                load_passkey_manifest(path)


class PasskeyArtifactTests(unittest.TestCase):
    def setUp(self):
        manifest = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json"
        )
        _, cases = expand_passkey_cases(manifest, FakeTokenizer(), source_state())
        self.case = cases[0]

    def test_validates_completed_case_identity_and_metrics(self):
        record = completed_record(self.case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / f"{record['case_id']}.json"
            path.parent.mkdir()
            path.write_text(json.dumps(record), encoding="utf-8")
            validated = validate_completed_passkey_case(root, record["case_id"])
            self.assertTrue(validated["generation"]["exact_match"])
            self.assertTrue(is_valid_completed_passkey_case(root, record["case_id"]))

    def test_rejects_inconsistent_exact_match(self):
        record = completed_record(self.case)
        record["generation"]["exact_match"] = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / f"{record['case_id']}.json"
            path.parent.mkdir()
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(PasskeyError, "exact_match is inconsistent"):
                validate_completed_passkey_case(root, record["case_id"])

    def test_summary_skips_invalid_stale_artifact_and_writes_csv(self):
        record = completed_record(self.case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases"
            cases.mkdir()
            (cases / f"{record['case_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            stale_id = "0" * 20
            (cases / f"{stale_id}.json").write_text("{}", encoding="utf-8")
            summary = aggregate_passkey_cases(
                root, expected_case_ids=[record["case_id"], stale_id]
            )
            self.assertEqual(len(summary), 1)
            self.assertTrue((root / "summary" / "cases.jsonl").is_file())
            self.assertTrue((root / "summary" / "cases.csv").is_file())


class PasskeyRunnerTests(unittest.TestCase):
    def test_smoke_quality_gate_requires_every_cell_to_match(self):
        manifest = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json"
        )
        records = [
            {
                "input": {"prompt_length": length, "position_percent": position},
                "generation": {"exact_match": True, "contains_target": True},
            }
            for length in manifest["prompt_lengths"]
            for position in manifest["passkey_positions_percent"]
        ]
        result = {"failure_records": 0}
        passkey_runner._attach_quality_counts(result, records)
        passkey_runner._attach_quality_gate(result, manifest, records)
        self.assertEqual(result["quality_gate"], "PASS")

        records[-1]["generation"]["exact_match"] = False
        result = {"failure_records": 0}
        passkey_runner._attach_quality_counts(result, records)
        passkey_runner._attach_quality_gate(result, manifest, records)
        self.assertEqual(result["quality_gate"], "FAIL")

    def test_calibration_quality_gate_requires_overall_and_per_cell_thresholds(self):
        manifest = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_calibration.json"
        )
        records = []
        for cell_index, (length, position) in enumerate(
            (length, position)
            for length in manifest["prompt_lengths"]
            for position in manifest["passkey_positions_percent"]
        ):
            matches = 5 if cell_index < 6 else 4
            records.extend(
                {
                    "input": {"prompt_length": length, "position_percent": position},
                    "generation": {
                        "exact_match": key_index < matches,
                        "contains_target": key_index < matches,
                    },
                }
                for key_index in range(5)
            )
        result = {"failure_records": 0}
        passkey_runner._attach_quality_counts(result, records)
        passkey_runner._attach_quality_gate(result, manifest, records)
        self.assertEqual(result["exact_matches"], 54)
        self.assertEqual(result["quality_gate"], "PASS")

        records[-2]["generation"]["exact_match"] = False
        result = {"failure_records": 0}
        passkey_runner._attach_quality_counts(result, records)
        passkey_runner._attach_quality_gate(result, manifest, records)
        self.assertEqual(result["quality_gate"], "FAIL")

    def test_completed_case_is_reused_without_loading_model(self):
        manifest = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json"
        )
        tokenizer = FakeTokenizer()
        _, cases = expand_passkey_cases(manifest, tokenizer, source_state())
        base_case = cases[0]
        record = completed_record(base_case)
        base_case = {**base_case, "case_id": record["case_id"]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {**manifest, "output_dir": str(root)}
            manifest_path = root / "input.json"
            manifest_path.write_text("{}", encoding="utf-8")
            cases_dir = root / "cases"
            cases_dir.mkdir()
            (cases_dir / f"{record['case_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            stale_failure = root / "failures" / f"{record['case_id']}.json"
            stale_failure.parent.mkdir()
            stale_failure.write_text("{}", encoding="utf-8")

            fake_config = type(
                "Config", (), {"model_type": "llama", "max_position_embeddings": 4096,
                                "rope_scaling": None}
            )()
            with (
                mock.patch.object(passkey_runner, "load_passkey_manifest", return_value=manifest),
                mock.patch.object(passkey_runner, "source_state_identity", return_value=source_state()),
                mock.patch.object(passkey_runner, "_load_preflight", return_value=(fake_config, tokenizer)),
                mock.patch.object(
                    passkey_runner,
                    "expand_passkey_cases",
                    return_value=([record["input"]["target"]], [base_case]),
                ),
                mock.patch.object(passkey_runner, "_load_model") as load_model,
            ):
                exit_code, result = passkey_runner.run_manifest(manifest_path)

            self.assertEqual(exit_code, passkey_runner.EXIT_SUCCESS)
            self.assertEqual(result["valid_reusable_cases"], 1)
            self.assertEqual(result["remaining_cases"], 0)
            self.assertEqual(result["completed_cases"], 1)
            self.assertEqual(result["failure_records"], 0)
            self.assertFalse(stale_failure.exists())
            quality = json.loads((root / "summary" / "quality.json").read_text())
            self.assertEqual(quality["quality_gate"], "INCOMPLETE")
            load_model.assert_not_called()

    def test_dirty_source_is_rejected_before_model_preflight(self):
        manifest = load_passkey_manifest(
            ROOT / "configs" / "cage_passkey_llama2_7b_fp16_smoke.json"
        )
        dirty = {**source_state(), "dirty": True, "dirty_sha256": "1" * 64}
        with (
            mock.patch.object(passkey_runner, "load_passkey_manifest", return_value=manifest),
            mock.patch.object(passkey_runner, "source_state_identity", return_value=dirty),
            mock.patch.object(passkey_runner, "_load_preflight") as load_preflight,
        ):
            exit_code, result = passkey_runner.run_manifest("manifest.json")
        self.assertEqual(exit_code, passkey_runner.EXIT_PREFLIGHT)
        self.assertIn("clean tracked source state", result["error"])
        load_preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
