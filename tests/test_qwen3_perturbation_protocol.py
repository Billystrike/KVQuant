import copy
import json
import unittest
from pathlib import Path


from utils.qwen3_perturbation_protocol import (
    LAYER_METRICS,
    Qwen3PerturbationError,
    aggregate_layer_metrics,
    load_perturbation_protocol,
    perturbation_case_id,
    validate_aggregates,
    validate_layer_records,
    validate_perturbation_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_memory_perturbation_protocol_v1.json"


def _layer_records():
    records = []
    for layer_idx in range(36):
        metrics = {name: float(layer_idx + 1) for name in LAYER_METRICS}
        metrics["topk_attention_overlap"] = layer_idx / 35
        records.append(
            {
                "layer_idx": layer_idx,
                "phase": "teacher_forced_decode",
                "query_source": "fp16_reference_final_position",
                "history_length": 1025,
                "metrics": metrics,
            }
        )
    return records


class Qwen3PerturbationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_sha256 = load_perturbation_protocol(PROTOCOL_PATH)

    def test_protocol_inherits_the_complete_quality_grid(self):
        grid = self.protocol["grid"]
        self.assertTrue(self.protocol["post_quality_design"])
        self.assertFalse(grid["quality_selected_subset"])
        self.assertEqual(grid["case_count"], 1300)
        self.assertEqual(grid["layer_record_count"], 1300 * 36)
        self.assertEqual(self.protocol["partitions"]["cage_qwen3"]["full_cases"], 1000)
        self.assertEqual(self.protocol["partitions"]["kitty_qwen3"]["full_cases"], 300)

    def test_measurement_matches_the_frozen_llama_local_definition(self):
        measurement = self.protocol["measurement"]
        self.assertEqual(measurement["continuation_token_index_used_as_decode_input"], 0)
        self.assertTrue(measurement["candidate_query_excluded_from_primary_metrics"])
        self.assertEqual(measurement["memory_estimate_length"], "prompt_length")
        self.assertEqual(
            self.protocol["metrics"]["primary"], "joint_post_o_proj_mse"
        )
        self.assertEqual(tuple(self.protocol["metrics"]["layer_metrics"]), LAYER_METRICS)

    def test_protocol_rejects_posthoc_subsetting(self):
        mutated = copy.deepcopy(self.protocol)
        mutated["grid"]["quality_selected_subset"] = True
        with self.assertRaises(Qwen3PerturbationError):
            validate_perturbation_protocol(mutated)

    def test_case_ids_bind_the_base_case_and_new_protocol(self):
        first = perturbation_case_id(
            base_case_id="abc", perturbation_protocol_sha256=self.protocol_sha256
        )
        second = perturbation_case_id(
            base_case_id="abc", perturbation_protocol_sha256=self.protocol_sha256
        )
        other = perturbation_case_id(
            base_case_id="def", perturbation_protocol_sha256=self.protocol_sha256
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 24)

    def test_layer_schema_and_aggregates_are_recomputed_exactly(self):
        records = _layer_records()
        validate_layer_records(records)
        aggregates = aggregate_layer_metrics(records)
        validate_aggregates(aggregates, records)
        self.assertEqual(aggregates["relative_k_reconstruction_error"]["mean"], 18.5)
        self.assertEqual(aggregates["relative_k_reconstruction_error"]["median"], 18.5)
        self.assertEqual(aggregates["relative_k_reconstruction_error"]["maximum"], 36.0)

    def test_layer_validation_rejects_missing_or_nonfinite_values(self):
        records = _layer_records()
        del records[0]["metrics"][LAYER_METRICS[0]]
        with self.assertRaises(Qwen3PerturbationError):
            validate_layer_records(records)
        records = _layer_records()
        records[0]["metrics"][LAYER_METRICS[0]] = float("nan")
        with self.assertRaises(Qwen3PerturbationError):
            validate_layer_records(records)


if __name__ == "__main__":
    unittest.main()
