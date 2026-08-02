from __future__ import annotations


def _attentions(model):
    return [layer.self_attn for layer in model.model.layers]


def begin_reference_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("reference")


def begin_candidate_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("candidate")


def collect_layer_metrics(model):
    records = []
    for layer_index, attention in enumerate(_attentions(model)):
        metrics = attention.pop_kv_experiment_metrics()
        if metrics is None:
            raise RuntimeError(f"layer {layer_index} did not emit KV metrics")
        records.append({"layer_index": layer_index, **metrics})
    return records


def reset_experiment_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("off")
        attention._kv_reference_query = None
        attention._kv_reference_key_history = None
        attention._kv_reference_value_history = None
        attention._kv_key_importance = None
        attention._kv_value_importance = None
        attention._kv_experiment_metrics = None


__all__ = [
    "begin_candidate_capture",
    "begin_reference_capture",
    "collect_layer_metrics",
    "reset_experiment_capture",
]
