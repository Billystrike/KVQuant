import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.cage_metrics import collect_cage_perturbation_metrics, compute_cage_perturbation_metrics
from utils.cage_experiment_schema import ExperimentPointError


class CageMetricsTest(unittest.TestCase):
    def _small_inputs(self):
        query_states = torch.tensor([[[[1.0, 0.0]]]])
        key_states = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]])
        value_states = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]]])
        key_states_hat = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]])
        value_states_hat = torch.tensor([[[[1.0, 1.0], [0.0, 1.0], [3.0, 1.0]]]])
        o_proj_weight = torch.eye(2)
        key_importance = torch.tensor([[2.0, 3.0]])
        value_importance = torch.tensor([[5.0, 7.0]])
        return {
            "query_states": query_states,
            "key_states": key_states,
            "key_states_hat": key_states_hat,
            "value_states": value_states,
            "value_states_hat": value_states_hat,
            "o_proj_weight": o_proj_weight,
            "key_importance": key_importance,
            "value_importance": value_importance,
        }

    def test_metrics_are_plain_json_serializable_floats(self):
        metrics = compute_cage_perturbation_metrics(**self._small_inputs())

        expected_keys = {
            "relative_k_reconstruction_error",
            "attention_logit_mse",
            "attention_score_kl",
            "topk_attention_overlap",
            "weighted_key_error",
            "relative_v_reconstruction_error",
            "attention_output_mse",
            "post_o_proj_mse",
            "weighted_value_error",
            "joint_attention_output_mse",
            "joint_post_o_proj_mse",
            "joint_attention_output_relative_error",
        }
        self.assertEqual(set(metrics), expected_keys)
        for value in metrics.values():
            self.assertIs(type(value), float)
        json.dumps(metrics, allow_nan=False)

    def test_weighted_channel_errors_use_importance_times_channel_mse(self):
        metrics = compute_cage_perturbation_metrics(**self._small_inputs())

        self.assertAlmostEqual(metrics["weighted_key_error"], 16.0 / 3.0)
        self.assertAlmostEqual(metrics["weighted_value_error"], 19.0 / 3.0)

    def test_attention_topk_overlap_defaults_to_ten_and_clamps_to_sequence_length(self):
        metrics = compute_cage_perturbation_metrics(**self._small_inputs())

        self.assertEqual(metrics["topk_attention_overlap"], 1.0)

    def test_attention_topk_overlap_can_use_smaller_k(self):
        metrics = compute_cage_perturbation_metrics(**self._small_inputs(), top_k=1)

        self.assertEqual(metrics["topk_attention_overlap"], 0.0)

    def test_identical_tensors_have_zero_error_and_full_topk_overlap(self):
        inputs = self._small_inputs()
        inputs["key_states_hat"] = inputs["key_states"]
        inputs["value_states_hat"] = inputs["value_states"]

        metrics = compute_cage_perturbation_metrics(**inputs)

        for key, value in metrics.items():
            expected = 1.0 if key == "topk_attention_overlap" else 0.0
            self.assertEqual(value, expected)

    def test_joint_metrics_are_zero_for_identical_cache(self):
        inputs = self._small_inputs()
        inputs["key_states_hat"] = inputs["key_states"]
        inputs["value_states_hat"] = inputs["value_states"]

        metrics = compute_cage_perturbation_metrics(**inputs)

        self.assertEqual(metrics["joint_attention_output_mse"], 0.0)
        self.assertEqual(metrics["joint_post_o_proj_mse"], 0.0)
        self.assertEqual(metrics["joint_attention_output_relative_error"], 0.0)

    def test_joint_metrics_respond_to_key_and_value_error(self):
        inputs = self._small_inputs()
        inputs["key_states_hat"] = inputs["key_states_hat"] + 0.25
        inputs["value_states_hat"] = inputs["value_states_hat"] - 0.25

        metrics = compute_cage_perturbation_metrics(**inputs)

        self.assertGreater(metrics["joint_attention_output_mse"], 0.0)
        self.assertGreater(metrics["joint_post_o_proj_mse"], 0.0)
        self.assertGreater(metrics["joint_attention_output_relative_error"], 0.0)

    def test_metrics_reject_non_finite_source_and_reconstruction_tensors(self):
        for field, value in (
            ("query_states", float("nan")),
            ("key_states", float("inf")),
            ("key_states_hat", float("nan")),
            ("value_states", float("-inf")),
            ("value_states_hat", float("inf")),
            ("o_proj_weight", float("nan")),
            ("key_importance", float("inf")),
            ("value_importance", float("nan")),
        ):
            with self.subTest(field=field):
                inputs = self._small_inputs()
                inputs[field] = inputs[field].clone()
                inputs[field].view(-1)[0] = value
                with self.assertRaisesRegex(ExperimentPointError, "non-finite"):
                    compute_cage_perturbation_metrics(**inputs)

    def test_collection_gate_skips_compute_and_dump_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(cage_collect_metrics=False, cage_dump_dir=tmpdir)

            metrics = collect_cage_perturbation_metrics(config, **self._small_inputs())

            self.assertIsNone(metrics)
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

    def test_collection_writes_jsonl_when_dump_dir_is_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(cage_collect_metrics=True, cage_dump_dir=tmpdir)

            metrics = collect_cage_perturbation_metrics(
                config,
                **self._small_inputs(),
                metadata={"phase": "prefill"},
            )

            output_path = Path(tmpdir) / "cage_perturbation_metrics.jsonl"
            self.assertTrue(output_path.exists())
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(record["phase"], "prefill")
            self.assertEqual(record["weighted_key_error"], metrics["weighted_key_error"])


if __name__ == "__main__":
    unittest.main()
