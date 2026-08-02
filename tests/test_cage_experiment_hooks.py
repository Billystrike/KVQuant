import types
import unittest

import torch

from utils.cage_experiment_hooks import (
    begin_candidate_capture,
    begin_reference_capture,
    collect_layer_metrics,
    reset_experiment_capture,
)


class _Attention:
    def __init__(self):
        self.phase = "off"
        self._kv_reference_query = None
        self._kv_reference_key_history = None
        self._kv_reference_value_history = None
        self._kv_key_importance = None
        self._kv_value_importance = None
        self._kv_experiment_metrics = None

    def set_kv_experiment_phase(self, phase):
        self.phase = phase

    def pop_kv_experiment_metrics(self):
        metrics = self._kv_experiment_metrics
        self._kv_experiment_metrics = None
        return metrics


def _model(*attentions):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=attention) for attention in attentions]
        )
    )


class CageExperimentHooksTest(unittest.TestCase):
    def test_begin_hooks_set_every_attention_phase(self):
        attentions = (_Attention(), _Attention())
        model = _model(*attentions)

        begin_reference_capture(model)
        self.assertEqual([attention.phase for attention in attentions], ["reference", "reference"])
        begin_candidate_capture(model)
        self.assertEqual([attention.phase for attention in attentions], ["candidate", "candidate"])

    def test_collect_adds_layer_index_and_consumes_metrics(self):
        attentions = (_Attention(), _Attention())
        attentions[0]._kv_experiment_metrics = {"score": 1.0}
        attentions[1]._kv_experiment_metrics = {"score": 2.0}

        records = collect_layer_metrics(_model(*attentions))

        self.assertEqual(records, [
            {"layer_index": 0, "score": 1.0},
            {"layer_index": 1, "score": 2.0},
        ])
        self.assertTrue(all(attention._kv_experiment_metrics is None for attention in attentions))

    def test_collect_rejects_layer_without_metrics(self):
        with self.assertRaisesRegex(RuntimeError, "layer 0 did not emit KV metrics"):
            collect_layer_metrics(_model(_Attention()))

    def test_reset_clears_all_retained_tensors(self):
        attention = _Attention()
        for name in (
            "_kv_reference_query",
            "_kv_reference_key_history",
            "_kv_reference_value_history",
            "_kv_key_importance",
            "_kv_value_importance",
            "_kv_experiment_metrics",
        ):
            setattr(attention, name, torch.ones(1, 1, 4096, 1))

        reset_experiment_capture(_model(attention))

        self.assertEqual(attention.phase, "off")
        for name in (
            "_kv_reference_query",
            "_kv_reference_key_history",
            "_kv_reference_value_history",
            "_kv_key_importance",
            "_kv_value_importance",
            "_kv_experiment_metrics",
        ):
            self.assertIsNone(getattr(attention, name))


if __name__ == "__main__":
    unittest.main()
