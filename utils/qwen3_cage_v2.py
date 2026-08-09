from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


ByteReport = dict[str, Any]


@dataclass(frozen=True)
class CageV2LayerOption:
    """One calibrated refinement choice for a single decoder layer."""

    one_bit_channels: int
    two_bit_channels: int
    packed_bytes: int
    expected_distortion: float
    label: str = ""

    def __post_init__(self) -> None:
        _require_nonnegative("one_bit_channels", self.one_bit_channels)
        _require_nonnegative("two_bit_channels", self.two_bit_channels)
        _require_positive("packed_bytes", self.packed_bytes)
        if not isinstance(self.expected_distortion, (int, float)) or not math.isfinite(
            float(self.expected_distortion)
        ):
            raise ValueError("expected_distortion must be finite")


def estimate_qwen3_cage_v2_bytes(
    *,
    seq_len: int,
    residual_length: int,
    one_bit_channels: int | Sequence[int] = 0,
    two_bit_channels: int | Sequence[int] = 16,
    sink_length: int = 0,
    key_base_group_size: int = 128,
    key_refinement_group_size: int = 128,
    value_group_size: int = 128,
    batch_size: int = 1,
    num_hidden_layers: int = 36,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
    bytes_per_channel_index: int = 1,
) -> ByteReport:
    """Return complete active packed paper-estimate bytes for CAGE-v2.

    The layout contains an all-channel INT2 K/V backbone and two disjoint Key
    refinement streams.  Selected channel indices are prompt-persistent and
    charged once per layer and KV head.  This is a logical packed estimate, not
    realized CUDA allocation.
    """

    _require_nonnegative("seq_len", seq_len)
    _require_nonnegative("sink_length", sink_length)
    for name, value in (
        ("residual_length", residual_length),
        ("key_base_group_size", key_base_group_size),
        ("key_refinement_group_size", key_refinement_group_size),
        ("value_group_size", value_group_size),
        ("batch_size", batch_size),
        ("num_hidden_layers", num_hidden_layers),
        ("num_key_value_heads", num_key_value_heads),
        ("head_dim", head_dim),
        ("bytes_per_meta", bytes_per_meta),
        ("bytes_per_full_precision", bytes_per_full_precision),
        ("bytes_per_channel_index", bytes_per_channel_index),
    ):
        _require_positive(name, value)

    one_counts = _normalize_layer_counts(
        "one_bit_channels", one_bit_channels, num_hidden_layers
    )
    two_counts = _normalize_layer_counts(
        "two_bit_channels", two_bit_channels, num_hidden_layers
    )
    for layer_idx, (one_count, two_count) in enumerate(zip(one_counts, two_counts)):
        if one_count + two_count > head_dim:
            raise ValueError(
                f"refinement counts exceed head_dim at layer {layer_idx}: "
                f"{one_count} + {two_count} > {head_dim}"
            )

    sink_tokens = min(seq_len, sink_length)
    non_sink_tokens = seq_len - sink_tokens
    key_quantized_tokens = non_sink_tokens - non_sink_tokens % residual_length
    key_residual_tokens = non_sink_tokens - key_quantized_tokens
    value_quantized_tokens = max(0, non_sink_tokens - residual_length)
    value_residual_tokens = non_sink_tokens - value_quantized_tokens

    layer_reports = []
    model_components: dict[str, int] = {}
    for layer_idx, (one_count, two_count) in enumerate(zip(one_counts, two_counts)):
        components = cage_v2_layer_bytes(
            key_quantized_tokens=key_quantized_tokens,
            value_quantized_tokens=value_quantized_tokens,
            key_residual_tokens=key_residual_tokens,
            value_residual_tokens=value_residual_tokens,
            sink_tokens=sink_tokens,
            one_bit_channels=one_count,
            two_bit_channels=two_count,
            key_base_group_size=key_base_group_size,
            key_refinement_group_size=key_refinement_group_size,
            value_group_size=value_group_size,
            batch_size=batch_size,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            bytes_per_meta=bytes_per_meta,
            bytes_per_full_precision=bytes_per_full_precision,
            bytes_per_channel_index=bytes_per_channel_index,
        )
        components["total_bytes"] = sum(components.values())
        layer_reports.append(
            {
                "layer_idx": layer_idx,
                "one_bit_channels_per_head": one_count,
                "two_bit_channels_per_head": two_count,
                "components": components,
            }
        )
        for name, value in components.items():
            model_components[name] = model_components.get(name, 0) + value

    return {
        "method": "cage-v2-sparse-refinement",
        "seq_len": seq_len,
        "residual_length": residual_length,
        "representation": "packed_paper_estimate",
        "token_state": {
            "sink_tokens": sink_tokens,
            "key_quantized_tokens": key_quantized_tokens,
            "key_fp16_tokens": sink_tokens + key_residual_tokens,
            "value_quantized_tokens": value_quantized_tokens,
            "value_fp16_tokens": sink_tokens + value_residual_tokens,
        },
        "policy": {
            "key_base_bits": 2,
            "value_bits": 2,
            "one_bit_channels_per_layer": list(one_counts),
            "two_bit_channels_per_layer": list(two_counts),
            "key_base_group_size": key_base_group_size,
            "key_refinement_group_size": key_refinement_group_size,
            "value_group_size": value_group_size,
            "sink_length": sink_length,
            "value_adaptive": False,
        },
        "layer_reports": layer_reports,
        "model_components": model_components,
        "model_total_bytes": model_components["total_bytes"],
    }


def cage_v2_layer_bytes(
    *,
    key_quantized_tokens: int,
    value_quantized_tokens: int,
    key_residual_tokens: int,
    value_residual_tokens: int,
    sink_tokens: int,
    one_bit_channels: int,
    two_bit_channels: int,
    key_base_group_size: int,
    key_refinement_group_size: int,
    value_group_size: int,
    batch_size: int,
    num_key_value_heads: int,
    head_dim: int,
    bytes_per_meta: int,
    bytes_per_full_precision: int,
    bytes_per_channel_index: int,
) -> dict[str, int]:
    """Return component bytes for one layer of the logical CAGE-v2 layout."""

    for name, value in (
        ("key_quantized_tokens", key_quantized_tokens),
        ("value_quantized_tokens", value_quantized_tokens),
        ("key_residual_tokens", key_residual_tokens),
        ("value_residual_tokens", value_residual_tokens),
        ("sink_tokens", sink_tokens),
        ("one_bit_channels", one_bit_channels),
        ("two_bit_channels", two_bit_channels),
    ):
        _require_nonnegative(name, value)
    if one_bit_channels + two_bit_channels > head_dim:
        raise ValueError("refinement counts exceed head_dim")

    values_per_token = batch_size * num_key_value_heads * head_dim
    key_base_meta = (
        batch_size
        * num_key_value_heads
        * head_dim
        * _ceil_div(key_quantized_tokens, key_base_group_size)
    )
    value_meta = (
        batch_size
        * num_key_value_heads
        * value_quantized_tokens
        * _ceil_div(head_dim, value_group_size)
    )
    refined_channels = one_bit_channels + two_bit_channels
    refinement_meta = (
        batch_size
        * num_key_value_heads
        * refined_channels
        * _ceil_div(key_quantized_tokens, key_refinement_group_size)
    )
    return {
        "key_base_payload_bytes": _packed_bytes(values_per_token * key_quantized_tokens, 2),
        "key_one_bit_refinement_payload_bytes": _packed_bytes(
            batch_size
            * num_key_value_heads
            * one_bit_channels
            * key_quantized_tokens,
            1,
        ),
        "key_two_bit_refinement_payload_bytes": _packed_bytes(
            batch_size
            * num_key_value_heads
            * two_bit_channels
            * key_quantized_tokens,
            2,
        ),
        "value_payload_bytes": _packed_bytes(values_per_token * value_quantized_tokens, 2),
        "key_base_scale_bytes": key_base_meta * bytes_per_meta,
        "key_base_zero_point_bytes": key_base_meta * bytes_per_meta,
        "key_refinement_scale_bytes": refinement_meta * bytes_per_meta,
        "key_refinement_zero_point_bytes": refinement_meta * bytes_per_meta,
        "value_scale_bytes": value_meta * bytes_per_meta,
        "value_zero_point_bytes": value_meta * bytes_per_meta,
        "key_refinement_index_bytes": (
            batch_size
            * num_key_value_heads
            * refined_channels
            * bytes_per_channel_index
        ),
        "sink_fp16_bytes": (
            2 * values_per_token * sink_tokens * bytes_per_full_precision
        ),
        "key_residual_fp16_bytes": (
            values_per_token * key_residual_tokens * bytes_per_full_precision
        ),
        "value_residual_fp16_bytes": (
            values_per_token * value_residual_tokens * bytes_per_full_precision
        ),
    }


def allocate_layer_options(
    options_by_layer: Sequence[Sequence[CageV2LayerOption]],
    *,
    target_bytes: int,
) -> dict[str, Any]:
    """Solve a deterministic multiple-choice packed-byte allocation.

    Exactly one calibrated option is selected for every layer.  Among feasible
    allocations the minimum summed expected distortion wins; ties prefer fewer
    bytes and then lexicographically smaller option labels.
    """

    _require_positive("target_bytes", target_bytes)
    if not options_by_layer:
        raise ValueError("options_by_layer must not be empty")
    normalized: list[tuple[CageV2LayerOption, ...]] = []
    for layer_idx, options in enumerate(options_by_layer):
        if not options:
            raise ValueError(f"layer {layer_idx} has no allocation options")
        normalized.append(tuple(options))

    # cost -> (error, selected option tuple)
    frontier: dict[int, tuple[float, tuple[CageV2LayerOption, ...]]] = {0: (0.0, ())}
    for options in normalized:
        expanded: dict[int, tuple[float, tuple[CageV2LayerOption, ...]]] = {}
        for prior_cost, (prior_error, prior_choices) in frontier.items():
            for option in options:
                cost = prior_cost + option.packed_bytes
                if cost > target_bytes:
                    continue
                candidate = (prior_error + float(option.expected_distortion), prior_choices + (option,))
                incumbent = expanded.get(cost)
                if incumbent is None or _allocation_key(candidate) < _allocation_key(incumbent):
                    expanded[cost] = candidate
        if not expanded:
            raise ValueError("target_bytes is smaller than every complete layer allocation")
        frontier = _prune_dominated_allocations(expanded)

    total_bytes, (expected_distortion, choices) = min(
        frontier.items(),
        key=lambda item: (
            item[1][0],
            item[0],
            tuple(option.label for option in item[1][1]),
        ),
    )
    return {
        "target_bytes": target_bytes,
        "total_bytes": total_bytes,
        "unused_bytes": target_bytes - total_bytes,
        "expected_distortion": expected_distortion,
        "one_bit_channels_per_layer": [option.one_bit_channels for option in choices],
        "two_bit_channels_per_layer": [option.two_bit_channels for option in choices],
        "selected_labels": [option.label for option in choices],
    }


def _prune_dominated_allocations(
    states: dict[int, tuple[float, tuple[CageV2LayerOption, ...]]]
) -> dict[int, tuple[float, tuple[CageV2LayerOption, ...]]]:
    kept: dict[int, tuple[float, tuple[CageV2LayerOption, ...]]] = {}
    best_error = math.inf
    for cost in sorted(states):
        state = states[cost]
        if state[0] < best_error:
            kept[cost] = state
            best_error = state[0]
    return kept


def _allocation_key(
    state: tuple[float, tuple[CageV2LayerOption, ...]]
) -> tuple[float, tuple[str, ...]]:
    return state[0], tuple(option.label for option in state[1])


def _normalize_layer_counts(
    name: str, value: int | Sequence[int], expected_layers: int
) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        _require_nonnegative(name, value)
        return (value,) * expected_layers
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer or a sequence")
    if len(value) != expected_layers:
        raise ValueError(f"{name} must contain {expected_layers} entries")
    normalized = tuple(value)
    for index, item in enumerate(normalized):
        _require_nonnegative(f"{name}[{index}]", item)
    return normalized


def _packed_bytes(num_values: int, bits: int) -> int:
    return _ceil_div(num_values * bits, 8)


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator) if numerator else 0


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


__all__ = [
    "CageV2LayerOption",
    "allocate_layer_options",
    "cage_v2_layer_bytes",
    "estimate_qwen3_cage_v2_bytes",
]
