import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_formal import (
    Qwen3FormalError,
    build_input_manifest,
    expand_partition_cases,
    formal_method_length_points,
    formal_scoring,
    load_execution_config,
    load_formal_protocol,
    validate_input_manifest,
    validate_completed_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_formal_quality_protocol_v1.json"
EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_formal_execution_v1.json"


class Qwen3FormalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha256 = load_formal_protocol(PROTOCOL_PATH)
        cls.token_ids = list(range(cls.protocol["input"]["token_count"]))
        from utils.qwen3_cases import token_ids_sha256

        cls.synthetic_protocol = copy.deepcopy(cls.protocol)
        cls.synthetic_protocol["input"]["token_ids_sha256"] = token_ids_sha256(
            cls.token_ids
        )
        cls.execution, cls.execution_sha256 = load_execution_config(
            EXECUTION_PATH,
            protocol=cls.protocol,
            protocol_sha256=cls.protocol_sha256,
        )

    def _manifest(self):
        return build_input_manifest(
            protocol=self.synthetic_protocol,
            protocol_sha256=self.protocol_sha256,
            token_ids=self.token_ids,
            corpus_snapshot_sha256=self.synthetic_protocol["input"][
                "corpus_snapshot_sha256"
            ],
            tokenizer_identity={"class": "synthetic"},
            source_state={"git_commit": "a" * 40, "dirty": False},
        )

    def test_builds_ordered_150_case_manifest(self):
        manifest = self._manifest()
        self.assertEqual(len(manifest["cases"]), 150)
        self.assertEqual(manifest["cases"][0]["input_case_id"], "qwen3-a00-l1024")
        self.assertEqual(manifest["cases"][-1]["input_case_id"], "qwen3-a49-l4032")
        self.assertEqual(len(manifest["cases"][0]["continuation_ids"]), 64)
        self.assertEqual(len(manifest["cases"][-1]["prompt_ids"]), 4032)

    def test_manifest_round_trip_validation(self):
        manifest = json.loads(json.dumps(self._manifest()))
        validate_input_manifest(
            manifest,
            protocol=self.synthetic_protocol,
            protocol_sha256=self.protocol_sha256,
        )

    def test_rejects_token_or_identity_mutation(self):
        manifest = self._manifest()
        manifest["cases"][0]["prompt_ids"][0] += 1
        with self.assertRaisesRegex(Qwen3FormalError, "token hash"):
            validate_input_manifest(
                manifest,
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
            )

        manifest = self._manifest()
        manifest["cases"][1]["input_case_id"] = manifest["cases"][0]["input_case_id"]
        with self.assertRaisesRegex(Qwen3FormalError, "ID/order"):
            validate_input_manifest(
                manifest,
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
            )

    def test_rejects_dirty_source_state(self):
        with self.assertRaisesRegex(Qwen3FormalError, "clean"):
            build_input_manifest(
                protocol=self.synthetic_protocol,
                protocol_sha256=self.protocol_sha256,
                token_ids=self.token_ids,
                corpus_snapshot_sha256=self.synthetic_protocol["input"][
                    "corpus_snapshot_sha256"
                ],
                tokenizer_identity={"class": "synthetic"},
                source_state={"git_commit": "a" * 40, "dirty": True},
            )

    def test_execution_points_freeze_partition_counts_and_controls(self):
        cage = formal_method_length_points(self.protocol, partition="cage_qwen3")
        kitty = formal_method_length_points(self.protocol, partition="kitty_qwen3")
        self.assertEqual(len(cage), 20)
        self.assertEqual(len(kitty), 6)
        identities = {(point["method_id"], point["prompt_length"]) for point in cage}
        self.assertIn(("cage-r288-fixed-uniform", 2048), identities)
        self.assertIn(("cage-r96-value-adaptive-only", 4032), identities)
        self.assertEqual(
            next(
                point for point in cage
                if point["method_id"] == "cage-r288-key-adaptive-only"
            )["config"]["value_importance"],
            "fixed_uniform",
        )

    def test_expands_exact_acceptance_and_full_partition_case_counts(self):
        manifest = self._manifest()
        cage_acceptance = expand_partition_cases(
            protocol=self.synthetic_protocol,
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            input_manifest=manifest,
            partition="cage_qwen3",
            stage="acceptance",
        )
        cage_full = expand_partition_cases(
            protocol=self.synthetic_protocol,
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            input_manifest=manifest,
            partition="cage_qwen3",
            stage="full",
        )
        kitty_acceptance = expand_partition_cases(
            protocol=self.synthetic_protocol,
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            input_manifest=manifest,
            partition="kitty_qwen3",
            stage="acceptance",
        )
        kitty_full = expand_partition_cases(
            protocol=self.synthetic_protocol,
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            input_manifest=manifest,
            partition="kitty_qwen3",
            stage="full",
        )
        self.assertEqual(len(cage_acceptance), 11)
        self.assertEqual(len(cage_full), 1000)
        self.assertEqual(len(kitty_acceptance), 3)
        self.assertEqual(len(kitty_full), 300)
        self.assertEqual({case["input"]["anchor_index"] for case in cage_acceptance}, {0})
        self.assertEqual(len({case["case_id"] for case in cage_full}), 1000)

    def test_formal_scoring_and_completed_result_validation(self):
        manifest = self._manifest()
        case = expand_partition_cases(
            protocol=self.synthetic_protocol,
            execution=self.execution,
            execution_sha256=self.execution_sha256,
            input_manifest=manifest,
            partition="cage_qwen3",
            stage="acceptance",
        )[0]
        scoring = formal_scoring([float(index) / 10 for index in range(64)])
        source_state = {"git_commit": "b" * 40, "dirty": False}
        record = {
            "schema_version": 1,
            "status": "completed",
            "case_id": case["case_id"],
            "partition": case["partition"],
            "method": case["method"],
            "input": case["input"],
            "identity": {
                "execution_id": self.execution["execution_id"],
                "execution_sha256": self.execution_sha256,
                "protocol_id": self.synthetic_protocol["protocol_id"],
                "protocol_sha256": self.protocol_sha256,
                "input_manifest_sha256": self.execution["input_manifest"]["sha256"],
                "source_state": source_state,
            },
            "scoring": scoring,
            "runtime": {
                "prefill_seconds": 1.0,
                "decode_seconds": 2.0,
                "elapsed_seconds": 3.0,
            },
            "cache": {"reported_seq_length": 1087},
            "model": {"class": "synthetic"},
        }
        validate_completed_result(
            record,
            expected_case=case,
            execution_id=self.execution["execution_id"],
            execution_sha256=self.execution_sha256,
            protocol_id=self.synthetic_protocol["protocol_id"],
            protocol_sha256=self.protocol_sha256,
            input_manifest_sha256=self.execution["input_manifest"]["sha256"],
            source_state=source_state,
        )
        record["scoring"]["mean_nll"] += 0.01
        with self.assertRaisesRegex(Qwen3FormalError, "mean_nll"):
            validate_completed_result(
                record,
                expected_case=case,
                execution_id=self.execution["execution_id"],
                execution_sha256=self.execution_sha256,
                protocol_id=self.synthetic_protocol["protocol_id"],
                protocol_sha256=self.protocol_sha256,
                input_manifest_sha256=self.execution["input_manifest"]["sha256"],
                source_state=source_state,
            )

    def test_formal_scoring_rejects_wrong_count_or_nonfinite(self):
        with self.assertRaises(Qwen3FormalError):
            formal_scoring([1.0] * 63)
        with self.assertRaises(Qwen3FormalError):
            formal_scoring([1.0] * 63 + [float("nan")])


if __name__ == "__main__":
    unittest.main()
