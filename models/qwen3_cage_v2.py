from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from models.cage_config import CageConfig
from models.cage_importance import compute_key_importance
from models.cage_quant import fake_quant_v_uniform
from models.cage_v2_quant import fake_quant_k_sparse_refinement
from models.qwen3_cage import Qwen3CageCache


LayerCounts = int | Sequence[int]


@dataclass(frozen=True)
class CageV2Config:
    """Accuracy-simulation policy for query-aware sparse Key refinement.

    Value storage is deliberately uniform INT2.  ``*_channels`` may be a
    model-wide integer or a per-layer sequence; counts are applied independently
    to every KV head in the corresponding layer.
    """

    one_bit_channels: LayerCounts = 0
    two_bit_channels: LayerCounts = 16
    key_base_group_size: int = 128
    key_refinement_group_size: int = 128
    value_group_size: int = 128
    key_base_clip_percentile: float = 1.0
    key_refinement_clip_percentile: float = 1.0
    sink_length: int = 0

    def __post_init__(self) -> None:
        for name in (
            "key_base_group_size",
            "key_refinement_group_size",
            "value_group_size",
        ):
            _require_positive_int(name, getattr(self, name))
        _require_nonnegative_int("sink_length", self.sink_length)
        for name in (
            "key_base_clip_percentile",
            "key_refinement_clip_percentile",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        _validate_layer_counts("one_bit_channels", self.one_bit_channels)
        _validate_layer_counts("two_bit_channels", self.two_bit_channels)

    def counts_for_layer(self, layer_idx: int, *, head_dim: int) -> tuple[int, int]:
        _require_nonnegative_int("layer_idx", layer_idx)
        _require_positive_int("head_dim", head_dim)
        one_bit = _layer_count(self.one_bit_channels, layer_idx, "one_bit_channels")
        two_bit = _layer_count(self.two_bit_channels, layer_idx, "two_bit_channels")
        if one_bit + two_bit > head_dim:
            raise ValueError(
                "CAGE-v2 refinement channel counts exceed head_dim: "
                f"{one_bit} + {two_bit} > {head_dim} at layer {layer_idx}"
            )
        return one_bit, two_bit


@dataclass(frozen=True)
class Qwen3CageV2LayerPolicy:
    one_bit_indices: torch.Tensor
    two_bit_indices: torch.Tensor
    one_bit_channels: int
    two_bit_channels: int


def select_sparse_refinement_channels(
    importance: torch.Tensor,
    *,
    one_bit_channels: int,
    two_bit_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select disjoint per-head refinement sets with stable tie handling."""

    if not isinstance(importance, torch.Tensor) or importance.ndim != 2:
        raise ValueError("importance must have shape [H_kv, D]")
    if not importance.is_floating_point() or not bool(torch.isfinite(importance).all()):
        raise ValueError("importance must be a finite floating point tensor")
    _require_nonnegative_int("one_bit_channels", one_bit_channels)
    _require_nonnegative_int("two_bit_channels", two_bit_channels)
    head_dim = importance.shape[1]
    if one_bit_channels + two_bit_channels > head_dim:
        raise ValueError("refinement channel counts exceed the channel dimension")

    ordered = torch.argsort(importance, dim=-1, descending=True, stable=True)
    two_bit = ordered[:, :two_bit_channels]
    one_bit = ordered[:, two_bit_channels : two_bit_channels + one_bit_channels]
    return one_bit, two_bit


class Qwen3CageV2Cache(Qwen3CageCache):
    """Qwen3 accuracy-simulation cache for CAGE-v2.

    The stored tensors are fake-quantized floating point tensors.  Packed-byte
    claims must use the separate estimator in :mod:`utils.qwen3_cage_v2`.
    """

    def __init__(
        self,
        config: CageV2Config,
        *,
        residual_length: int,
        require_batch_size_one: bool = True,
    ) -> None:
        if not isinstance(config, CageV2Config):
            raise TypeError("config must be a CageV2Config")
        _require_positive_int("residual_length", residual_length)
        bridge = CageConfig(
            cage_enable=True,
            cage_mode="fake",
            cage_k_enable=True,
            cage_v_enable=True,
            cage_k_group_sizes=[config.key_base_group_size],
            cage_k_clip_percentiles=[config.key_base_clip_percentile],
            cage_k_num_buckets=1,
            cage_v_group_sizes=[config.value_group_size],
            cage_v_clip_percentiles=[1.0],
            cage_v_num_buckets=1,
        )
        super().__init__(
            bridge,
            residual_length=residual_length,
            bits=2,
            require_batch_size_one=require_batch_size_one,
        )
        self.v2_config = config

    def _build_layer_policy(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        o_proj_weight: torch.Tensor,
        layer_idx: int,
    ) -> Qwen3CageV2LayerPolicy:
        del value_states, o_proj_weight
        num_key_value_groups = query_states.shape[1] // key_states.shape[1]
        importance = compute_key_importance(
            query_states,
            key_states,
            num_key_value_groups=num_key_value_groups,
        )
        one_count, two_count = self.v2_config.counts_for_layer(
            layer_idx,
            head_dim=key_states.shape[-1],
        )
        one_bit, two_bit = select_sparse_refinement_channels(
            importance,
            one_bit_channels=one_count,
            two_bit_channels=two_count,
        )
        return Qwen3CageV2LayerPolicy(
            one_bit_indices=one_bit.to(device=key_states.device),
            two_bit_indices=two_bit.to(device=key_states.device),
            one_bit_channels=one_count,
            two_bit_channels=two_count,
        )

    def _quantize_new_key_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
        policy: Qwen3CageV2LayerPolicy,
    ) -> tuple[torch.Tensor, int]:
        sink_tokens = min(states.shape[-2], self.v2_config.sink_length)
        non_sink_length = states.shape[-2] - sink_tokens
        quantized_non_sink = non_sink_length - non_sink_length % self.residual_length
        new_quantized_end = sink_tokens + quantized_non_sink
        start = max(old_quantized_length, sink_tokens)
        if new_quantized_end > start:
            states[:, :, start:new_quantized_end, :] = fake_quant_k_sparse_refinement(
                states[:, :, start:new_quantized_end, :].contiguous(),
                one_bit_indices=policy.one_bit_indices,
                two_bit_indices=policy.two_bit_indices,
                base_group_size=self.v2_config.key_base_group_size,
                refinement_group_size=self.v2_config.key_refinement_group_size,
                base_clip_percentile=self.v2_config.key_base_clip_percentile,
                refinement_clip_percentile=self.v2_config.key_refinement_clip_percentile,
            )
        return states, new_quantized_end

    def _quantize_new_value_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
        policy: Qwen3CageV2LayerPolicy,
    ) -> tuple[torch.Tensor, int]:
        del policy
        sink_tokens = min(states.shape[-2], self.v2_config.sink_length)
        new_quantized_end = max(sink_tokens, states.shape[-2] - self.residual_length)
        start = max(old_quantized_length, sink_tokens)
        if new_quantized_end > start:
            states[:, :, start:new_quantized_end, :] = fake_quant_v_uniform(
                states[:, :, start:new_quantized_end, :].contiguous(),
                group_size=self.v2_config.value_group_size,
                bits=2,
            )
        return states, new_quantized_end


def _layer_count(value: LayerCounts, layer_idx: int, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if layer_idx >= len(value):
        raise ValueError(f"{name} has no entry for layer {layer_idx}")
    return int(value[layer_idx])


def _validate_layer_counts(name: str, value: LayerCounts) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        _require_nonnegative_int(name, value)
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonnegative integer or a nonempty sequence")
    for index, item in enumerate(value):
        _require_nonnegative_int(f"{name}[{index}]", item)


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


__all__ = [
    "CageV2Config",
    "Qwen3CageV2Cache",
    "Qwen3CageV2LayerPolicy",
    "select_sparse_refinement_channels",
]
