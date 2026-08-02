import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import scripts.cage_run_ppl as ppl_runner
from utils.cage_ppl import (
    PPL_CONTINUATION_TOKENS,
    PPL_PAIRED_ANCHOR_INDICES,
    PPL_PAIRED_SELECTION_ID,
    PPL_SCHEMA_VERSION,
    PPLError,
    aggregate_ppl_cases,
    continuation_anchor,
    expand_ppl_cases,
    load_ppl_manifest,
    token_ids_sha256,
    validate_completed_ppl_case,
    validate_corpus_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def source_state():
    return {
        "git_commit": "abc",
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


class FakeTokenizer:
    vocab_size = 128
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        values = [3 + (ord(character) % 61) for character in text]
        if add_special_tokens:
            values.insert(0, self.bos_token_id)
        return {"input_ids": values}


class SyntheticTokenStream:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if not isinstance(item, slice):
            return item % 100
        start, stop, step = item.indices(self.length)
        return [index % 100 for index in range(start, stop, step)]


def small_manifest(methods=None):
    token_ids = [index % 100 for index in range(5000)]
    return {
        "model": {
            "reference": "/model",
            "dtype": "float16",
            "device": "cpu",
            "max_position_embeddings": 4096,
        },
        "corpus": {
            "id": "fixture",
            "config": "fixture",
            "split": "test",
            "revision": "rev",
            "expected_token_count": len(token_ids),
            "expected_token_ids_sha256": token_ids_sha256(token_ids),
        },
        "methods": methods or [{
            "id": "fp16",
            "method": "fp16",
            "method_config": {},
        }],
        "prompt_lengths": [512],
        "anchor_indices": [0],
        "measurement": {
            "selection_id": "interior-sixths-v1",
            "continuation_tokens": 64,
        },
        "output_dir": "/output",
        "protocol_stage": "fixture",
    }, token_ids


def completed_record(case, *, method_name=None):
    method = case["method"]
    name = method_name or method["name"]
    nlls = [1.0] * PPL_CONTINUATION_TOKENS
    reference = None
    if name == "fp16":
        reference = {
            "token_nlls": list(nlls),
            "nll_sum": 64.0,
            "mean_nll": 1.0,
            "perplexity": math.e,
            "mean_absolute_token_nll_delta": 0.0,
            "max_absolute_token_nll_delta": 0.0,
        }
    return {
        "schema_version": PPL_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "status": "completed",
        "model": {
            "reference": "/model",
            "dtype": "float16",
            "device": "cpu",
            "max_position_embeddings": 4096,
            "model_type": "llama",
        },
        "method": {**method, "name": name},
        "input": case["input"],
        "scoring": {
            "primary_metric": "cache_dependent_decode_nll",
            "boundary_target_count": 1,
            "decode_target_count": 63,
            "all_target_count": 64,
            "token_nlls": nlls,
            "boundary_nll": 1.0,
            "decode_nll_sum": 63.0,
            "decode_mean_nll": 1.0,
            "decode_perplexity": math.e,
            "all_nll_sum": 64.0,
            "all_mean_nll": 1.0,
            "all_perplexity": math.e,
            "fp16_one_shot_reference": reference,
        },
        "runtime_diagnostics": {
            "load_seconds": 1.0,
            "prefill_seconds": 1.0,
            "decode_seconds": 1.0,
            "reference_seconds": 1.0 if name == "fp16" else 0.0,
            "elapsed_seconds": 3.0,
            "cuda_max_allocated_bytes": 0,
            "cuda_max_reserved_bytes": 0,
        },
        "provenance": {"source_state": source_state()},
    }


class PPLManifestTests(unittest.TestCase):
    def test_checked_in_acceptance_and_full_matrices(self):
        acceptance = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_llama2_7b_acceptance.json"
        )
        full = load_ppl_manifest(ROOT / "configs" / "cage_ppl_llama2_7b.json")
        self.assertEqual(acceptance["protocol_stage"], "acceptance")
        self.assertEqual(full["protocol_stage"], "full")
        self.assertEqual(len(acceptance["methods"]), 3)
        self.assertEqual(len(full["methods"]), 10)
        self.assertEqual(
            len(acceptance["methods"])
            * len(acceptance["prompt_lengths"])
            * len(acceptance["anchor_indices"]),
            6,
        )
        self.assertEqual(
            len(full["methods"])
            * len(full["prompt_lengths"])
            * len(full["anchor_indices"]),
            200,
        )
        self.assertEqual(acceptance["output_dir"], full["output_dir"])

        stream = SyntheticTokenStream(full["corpus"]["expected_token_count"])
        acceptance_cases = expand_ppl_cases(acceptance, stream, source_state())
        full_cases = expand_ppl_cases(full, stream, source_state())
        self.assertEqual(len(acceptance_cases), 6)
        self.assertEqual(len(full_cases), 200)
        self.assertTrue(
            {case["case_id"] for case in acceptance_cases}.issubset(
                {case["case_id"] for case in full_cases}
            )
        )

    def test_checked_in_paired_acceptance_is_subset_of_fifty_anchor_matrix(self):
        acceptance = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_paired_llama2_7b_acceptance.json"
        )
        full = load_ppl_manifest(ROOT / "configs" / "cage_ppl_paired_llama2_7b.json")
        self.assertEqual(acceptance["protocol_stage"], "paired_acceptance")
        self.assertEqual(full["protocol_stage"], "paired_full")
        self.assertEqual(len(acceptance["methods"]), 5)
        self.assertEqual(len(full["methods"]), 5)
        self.assertEqual(len(full["anchor_indices"]), 50)
        self.assertEqual(
            len(acceptance["methods"])
            * len(acceptance["prompt_lengths"])
            * len(acceptance["anchor_indices"]),
            10,
        )
        self.assertEqual(
            len(full["methods"])
            * len(full["prompt_lengths"])
            * len(full["anchor_indices"]),
            1000,
        )
        self.assertEqual(acceptance["output_dir"], full["output_dir"])

        stream = SyntheticTokenStream(full["corpus"]["expected_token_count"])
        acceptance_cases = expand_ppl_cases(acceptance, stream, source_state())
        full_cases = expand_ppl_cases(full, stream, source_state())
        self.assertEqual(len(acceptance_cases), 10)
        self.assertEqual(len(full_cases), 1000)
        self.assertTrue(
            {case["case_id"] for case in acceptance_cases}.issubset(
                {case["case_id"] for case in full_cases}
            )
        )

    def test_checked_in_mechanism_ablation_is_frozen_and_acceptance_is_subset(self):
        acceptance = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_mechanism_ablation_llama2_7b_acceptance.json"
        )
        full = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_mechanism_ablation_llama2_7b.json"
        )
        self.assertEqual(
            acceptance["protocol_stage"], "mechanism_ablation_acceptance"
        )
        self.assertEqual(full["protocol_stage"], "mechanism_ablation_full")
        self.assertEqual(len(acceptance["methods"]), 10)
        self.assertEqual(len(full["methods"]), 10)
        self.assertEqual(len(full["anchor_indices"]), 50)
        self.assertTrue(all(
            method["method"] == "cage"
            and method["method_config"]["cage_ablation"] is True
            for method in full["methods"]
        ))
        self.assertEqual(
            len(acceptance["methods"])
            * len(acceptance["prompt_lengths"])
            * len(acceptance["anchor_indices"]),
            20,
        )
        self.assertEqual(
            len(full["methods"])
            * len(full["prompt_lengths"])
            * len(full["anchor_indices"]),
            2000,
        )
        self.assertEqual(acceptance["output_dir"], full["output_dir"])

        stream = SyntheticTokenStream(full["corpus"]["expected_token_count"])
        acceptance_cases = expand_ppl_cases(acceptance, stream, source_state())
        full_cases = expand_ppl_cases(full, stream, source_state())
        self.assertEqual(len(acceptance_cases), 20)
        self.assertEqual(len(full_cases), 2000)
        self.assertTrue(
            {case["case_id"] for case in acceptance_cases}.issubset(
                {case["case_id"] for case in full_cases}
            )
        )

    def test_rejects_corpus_identity_drift(self):
        raw = json.loads(
            (ROOT / "configs" / "cage_ppl_llama2_7b_acceptance.json").read_text()
        )
        raw["corpus"]["expected_token_count"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(PPLError, "frozen WikiText"):
                load_ppl_manifest(path)


class PPLInputTests(unittest.TestCase):
    def test_anchor_selection_and_nested_prompt_hashes_are_stable(self):
        manifest, token_ids = small_manifest()
        expected = 4032 + ((5000 - 64 - 4032) // 6)
        self.assertEqual(continuation_anchor(5000, 0), expected)
        cases = expand_ppl_cases(manifest, token_ids, source_state())
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(len(case["prompt_ids"]), 512)
        self.assertEqual(len(case["continuation_ids"]), 64)
        self.assertEqual(case["input"]["continuation_start"], expected)
        self.assertEqual(case["input"]["prompt_ids_sha256"], token_ids_sha256(case["prompt_ids"]))

    def test_fifty_anchor_grid_is_ordered_and_nonoverlapping_at_max_context(self):
        token_count = 341468
        starts = [
            continuation_anchor(
                token_count,
                anchor_index,
                selection_id=PPL_PAIRED_SELECTION_ID,
            )
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES
        ]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[0], 4032)
        self.assertLessEqual(starts[-1] + PPL_CONTINUATION_TOKENS, token_count)
        self.assertTrue(
            all(right - left > 4032 for left, right in zip(starts, starts[1:]))
        )

    def test_validates_corpus_content_and_token_identity(self):
        texts = ["ab", "cd"]
        tokenizer = FakeTokenizer()
        joined = "ab\n\ncd"
        token_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
        manifest = {
            "corpus": {
                "join_separator": "\n\n",
                "add_special_tokens": False,
                "expected_rows": 2,
                "expected_nonempty_rows": 2,
                "expected_joined_utf8_bytes": len(joined.encode()),
                "expected_joined_text_sha256": __import__("hashlib").sha256(joined.encode()).hexdigest(),
                "expected_token_count": len(token_ids),
                "expected_token_ids_sha256": token_ids_sha256(token_ids),
            }
        }
        snapshot, actual_ids = validate_corpus_snapshot(
            manifest,
            texts,
            tokenizer,
            dataset_fingerprint="fingerprint",
            datasets_version="5.0.0",
        )
        self.assertEqual(actual_ids, token_ids)
        self.assertEqual(snapshot["dataset_fingerprint"], "fingerprint")
        manifest["corpus"]["expected_rows"] = 3
        with self.assertRaisesRegex(PPLError, "identity mismatch"):
            validate_corpus_snapshot(
                manifest, texts, tokenizer,
                dataset_fingerprint="fingerprint", datasets_version="5.0.0",
            )


class PPLScoringTests(unittest.TestCase):
    def test_primary_score_excludes_boundary_target(self):
        values = [10.0] + [1.0] * 63
        result = ppl_runner._score_from_nlls(values, values)
        self.assertEqual(result["boundary_nll"], 10.0)
        self.assertEqual(result["decode_target_count"], 63)
        self.assertEqual(result["decode_mean_nll"], 1.0)
        self.assertAlmostEqual(result["decode_perplexity"], math.e)
        self.assertEqual(
            result["fp16_one_shot_reference"]["max_absolute_token_nll_delta"],
            0.0,
        )

    def test_run_case_uses_prefill_then_63_single_token_decode_calls(self):
        class FakeModel:
            config = SimpleNamespace(model_type="llama")

            def __init__(self):
                self.calls = []

            def __call__(self, *, input_ids, past_key_values=None, use_cache, return_dict):
                self.calls.append((input_ids.shape[-1], past_key_values is not None, use_cache))
                logits = torch.zeros((1, input_ids.shape[-1], 128), dtype=torch.float32)
                past = (len(self.calls),) if use_cache else None
                return SimpleNamespace(logits=logits, past_key_values=past)

        manifest, token_ids = small_manifest()
        cases = expand_ppl_cases(manifest, token_ids, source_state())
        model = FakeModel()
        record = ppl_runner.run_case(
            case=cases[0], manifest=manifest, model=model,
            provenance={"source_state": source_state()}, load_seconds=0.5,
        )
        self.assertEqual(len(model.calls), 65)
        self.assertEqual(model.calls[0], (576, False, False))
        self.assertEqual(model.calls[1], (512, False, True))
        self.assertTrue(all(call == (1, True, True) for call in model.calls[2:]))
        self.assertEqual(record["scoring"]["decode_target_count"], 63)
        self.assertAlmostEqual(record["scoring"]["decode_mean_nll"], math.log(128), places=5)


class PPLArtifactTests(unittest.TestCase):
    def setUp(self):
        manifest, token_ids = small_manifest()
        self.case = expand_ppl_cases(manifest, token_ids, source_state())[0]

    def test_validates_completed_case_and_recomputes_identity(self):
        record = completed_record(self.case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / f"{record['case_id']}.json"
            path.parent.mkdir()
            path.write_text(json.dumps(record), encoding="utf-8")
            validated = validate_completed_ppl_case(root, record["case_id"])
            self.assertEqual(validated["scoring"]["decode_target_count"], 63)

    def test_rejects_inconsistent_aggregate(self):
        record = completed_record(self.case)
        record["scoring"]["decode_nll_sum"] = 62.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / f"{record['case_id']}.json"
            path.parent.mkdir()
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(PPLError, "inconsistent"):
                validate_completed_ppl_case(root, record["case_id"])

    def test_aggregation_is_limited_to_expected_ids(self):
        record = completed_record(self.case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases"
            cases.mkdir()
            (cases / f"{record['case_id']}.json").write_text(json.dumps(record), encoding="utf-8")
            (cases / f"{'0' * 20}.json").write_text("{}", encoding="utf-8")
            rows = aggregate_ppl_cases(root, expected_case_ids=[record["case_id"]])
            self.assertEqual(len(rows), 1)
            self.assertEqual(len((root / "summary" / "cases.jsonl").read_text().splitlines()), 1)


class PPLRunnerTests(unittest.TestCase):
    def test_dirty_source_is_rejected_before_dataset_preflight(self):
        manifest = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_llama2_7b_acceptance.json"
        )
        dirty = {**source_state(), "dirty": True, "dirty_sha256": "1" * 64}
        with (
            mock.patch.object(ppl_runner, "load_ppl_manifest", return_value=manifest),
            mock.patch.object(ppl_runner, "source_state_identity", return_value=dirty),
            mock.patch.object(ppl_runner, "_load_preflight") as preflight,
        ):
            exit_code, result = ppl_runner.run_manifest("manifest.json")
        self.assertEqual(exit_code, ppl_runner.EXIT_PREFLIGHT)
        self.assertIn("clean tracked source state", result["error"])
        preflight.assert_not_called()

    def test_acceptance_loads_each_method_once_and_completes_six_cases(self):
        manifest = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_llama2_7b_acceptance.json"
        )
        stream = SyntheticTokenStream(manifest["corpus"]["expected_token_count"])
        cases = expand_ppl_cases(manifest, stream, source_state())
        native_config = SimpleNamespace(
            model_type="llama", max_position_embeddings=4096, rope_scaling=None
        )
        snapshot = {
            "token_count": manifest["corpus"]["expected_token_count"],
            "token_ids_sha256": manifest["corpus"]["expected_token_ids_sha256"],
        }

        def fake_run_case(**kwargs):
            record = completed_record(kwargs["case"])
            record["model"] = {
                **manifest["model"],
                "model_type": "llama",
            }
            return record

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            selected_manifest = {**manifest, "output_dir": str(output)}
            with (
                mock.patch.object(
                    ppl_runner, "load_ppl_manifest", return_value=selected_manifest
                ),
                mock.patch.object(
                    ppl_runner, "source_state_identity", return_value=source_state()
                ),
                mock.patch.object(
                    ppl_runner,
                    "_load_preflight",
                    return_value=(native_config, FakeTokenizer(), snapshot, stream),
                ),
                mock.patch.object(
                    ppl_runner, "_load_model", return_value=(object(), 0.1)
                ) as load_model,
                mock.patch.object(
                    ppl_runner, "run_case", side_effect=fake_run_case
                ) as run_case,
                mock.patch.object(
                    ppl_runner,
                    "collect_provenance",
                    return_value={"source_state": source_state()},
                ),
                mock.patch.object(ppl_runner.gc, "collect"),
                mock.patch.object(ppl_runner.torch.cuda, "empty_cache"),
            ):
                exit_code, result = ppl_runner.run_manifest("manifest.json")

        self.assertEqual(exit_code, ppl_runner.EXIT_SUCCESS)
        self.assertEqual(load_model.call_count, 3)
        self.assertEqual(run_case.call_count, 6)
        self.assertEqual(result["completed_cases"], 6)
        self.assertEqual(result["failure_records"], 0)
        self.assertEqual(result["completion_gate"], "PASS")

    def test_paired_acceptance_loads_five_methods_once_and_completes_ten_cases(self):
        manifest = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_paired_llama2_7b_acceptance.json"
        )
        stream = SyntheticTokenStream(manifest["corpus"]["expected_token_count"])
        native_config = SimpleNamespace(
            model_type="llama", max_position_embeddings=4096, rope_scaling=None
        )
        snapshot = {
            "token_count": manifest["corpus"]["expected_token_count"],
            "token_ids_sha256": manifest["corpus"]["expected_token_ids_sha256"],
        }

        def fake_run_case(**kwargs):
            record = completed_record(kwargs["case"])
            record["model"] = {**manifest["model"], "model_type": "llama"}
            return record

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            selected_manifest = {**manifest, "output_dir": str(output)}
            with (
                mock.patch.object(
                    ppl_runner, "load_ppl_manifest", return_value=selected_manifest
                ),
                mock.patch.object(
                    ppl_runner, "source_state_identity", return_value=source_state()
                ),
                mock.patch.object(
                    ppl_runner,
                    "_load_preflight",
                    return_value=(native_config, FakeTokenizer(), snapshot, stream),
                ),
                mock.patch.object(
                    ppl_runner, "_load_model", return_value=(object(), 0.1)
                ) as load_model,
                mock.patch.object(
                    ppl_runner, "run_case", side_effect=fake_run_case
                ) as run_case,
                mock.patch.object(
                    ppl_runner,
                    "collect_provenance",
                    return_value={"source_state": source_state()},
                ),
                mock.patch.object(ppl_runner.gc, "collect"),
                mock.patch.object(ppl_runner.torch.cuda, "empty_cache"),
            ):
                exit_code, result = ppl_runner.run_manifest("manifest.json")

        self.assertEqual(exit_code, ppl_runner.EXIT_SUCCESS)
        self.assertEqual(load_model.call_count, 5)
        self.assertEqual(run_case.call_count, 10)
        self.assertEqual(result["protocol_stage"], "paired_acceptance")
        self.assertEqual(result["completed_cases"], 10)
        self.assertEqual(result["failure_records"], 0)
        self.assertEqual(result["completion_gate"], "PASS")

    def test_mechanism_ablation_acceptance_loads_ten_cage_methods(self):
        manifest = load_ppl_manifest(
            ROOT / "configs" / "cage_ppl_mechanism_ablation_llama2_7b_acceptance.json"
        )
        stream = SyntheticTokenStream(manifest["corpus"]["expected_token_count"])
        native_config = SimpleNamespace(
            model_type="llama", max_position_embeddings=4096, rope_scaling=None
        )
        snapshot = {
            "token_count": manifest["corpus"]["expected_token_count"],
            "token_ids_sha256": manifest["corpus"]["expected_token_ids_sha256"],
        }

        def fake_run_case(**kwargs):
            record = completed_record(kwargs["case"])
            record["model"] = {**manifest["model"], "model_type": "llama"}
            return record

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            selected_manifest = {**manifest, "output_dir": str(output)}
            with (
                mock.patch.object(
                    ppl_runner, "load_ppl_manifest", return_value=selected_manifest
                ),
                mock.patch.object(
                    ppl_runner, "source_state_identity", return_value=source_state()
                ),
                mock.patch.object(
                    ppl_runner,
                    "_load_preflight",
                    return_value=(native_config, FakeTokenizer(), snapshot, stream),
                ),
                mock.patch.object(
                    ppl_runner, "_load_model", return_value=(object(), 0.1)
                ) as load_model,
                mock.patch.object(
                    ppl_runner, "run_case", side_effect=fake_run_case
                ) as run_case,
                mock.patch.object(
                    ppl_runner,
                    "collect_provenance",
                    return_value={"source_state": source_state()},
                ),
                mock.patch.object(ppl_runner.gc, "collect"),
                mock.patch.object(ppl_runner.torch.cuda, "empty_cache"),
            ):
                exit_code, result = ppl_runner.run_manifest("manifest.json")

        self.assertEqual(exit_code, ppl_runner.EXIT_SUCCESS)
        self.assertEqual(load_model.call_count, 10)
        self.assertEqual(run_case.call_count, 20)
        self.assertEqual(result["protocol_stage"], "mechanism_ablation_acceptance")
        self.assertEqual(result["completed_cases"], 20)
        self.assertEqual(result["failure_records"], 0)
        self.assertEqual(result["completion_gate"], "PASS")
        self.assertEqual(result["fp16_calibration_cases"], 0)
        self.assertIsNone(result["fp16_calibration_max_mean_abs_token_nll_delta"])
        self.assertIsNone(result["fp16_calibration_max_abs_token_nll_delta"])


if __name__ == "__main__":
    unittest.main()
