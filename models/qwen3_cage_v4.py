from __future__ import annotations

from dataclasses import dataclass

import torch

from models.qwen3_cage_v2 import (
    CageV2Config,
    Qwen3CageV2Cache,
    Qwen3CageV2LayerPolicy,
    select_sparse_refinement_channels,
)


DTQI_GLOBAL_QUERY_WEIGHT = 0.5
DTQI_RECENT_QUERY_WEIGHT = 0.5


@dataclass(frozen=True)
class CageV4DTQIConfig(CageV2Config):
    """CAGE-v3 storage with the frozen CAGE-v4-DTQI ranking policy.

    The recent Query window is not a configuration field: it is always the
    cache's already-frozen residual length.  DTQI therefore adds no searched
    window hyperparameter and does not change the packed representation.
    """


@dataclass(frozen=True)
class Qwen3CageV4DTQILayerPolicy(Qwen3CageV2LayerPolicy):
    importance_policy: str = "dual_timescale_query_energy_times_key_variance"
    global_query_weight: float = DTQI_GLOBAL_QUERY_WEIGHT
    recent_query_weight: float = DTQI_RECENT_QUERY_WEIGHT


def compute_dtqi_key_importance(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    num_key_value_groups: int,
    recent_query_window: int,
) -> torch.Tensor:
    """Return the frozen dual-timescale Query-energy Key-channel score.

    ``recent_query_window`` is supplied by ``Qwen3CageV4DTQICache`` only as
    its frozen residual length.  Query and Key tensors are the post-RoPE
    prefill tensors shaped ``[B, H, T, D]``.  The result is ``[H_kv, D]``.
    """

    _validate_dtqi_inputs(
        query_states,
        key_states,
        num_key_value_groups=num_key_value_groups,
        recent_query_window=recent_query_window,
    )
    batch_size, _, sequence_length, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[1]
    recent_length = min(sequence_length, recent_query_window)
    global_energy = query_states.square().mean(dim=2)
    recent_energy = query_states[:, :, -recent_length:, :].square().mean(dim=2)
    query_energy = (
        DTQI_GLOBAL_QUERY_WEIGHT * global_energy
        + DTQI_RECENT_QUERY_WEIGHT * recent_energy
    )
    if num_key_value_groups > 1:
        query_energy = query_energy.view(
            batch_size,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        ).sum(dim=2)
    key_variance = key_states.var(dim=2, unbiased=False)
    importance = (query_energy * key_variance).mean(dim=0)
    if not bool(torch.isfinite(importance).all()):
        raise ValueError("CAGE-v4-DTQI importance must be finite")
    return importance


class Qwen3CageV4DTQICache(Qwen3CageV2Cache):
    """Accuracy simulation for the single frozen CAGE-v4-DTQI candidate."""

    def __init__(
        self,
        config: CageV4DTQIConfig,
        *,
        residual_length: int,
        require_batch_size_one: bool = True,
    ) -> None:
        if not isinstance(config, CageV4DTQIConfig):
            raise TypeError("config must be a CageV4DTQIConfig")
        super().__init__(
            config,
            residual_length=residual_length,
            require_batch_size_one=require_batch_size_one,
        )
        self.dtqi_config = config

    def _build_layer_policy(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        o_proj_weight: torch.Tensor,
        layer_idx: int,
    ) -> Qwen3CageV4DTQILayerPolicy:
        del value_states, o_proj_weight
        num_key_value_groups = query_states.shape[1] // key_states.shape[1]
        importance = compute_dtqi_key_importance(
            query_states,
            key_states,
            num_key_value_groups=num_key_value_groups,
            recent_query_window=self.residual_length,
        )
        one_count, two_count = self.dtqi_config.counts_for_layer(
            layer_idx,
            head_dim=key_states.shape[-1],
        )
        one_bit, two_bit = select_sparse_refinement_channels(
            importance,
            one_bit_channels=one_count,
            two_bit_channels=two_count,
        )
        return Qwen3CageV4DTQILayerPolicy(
            one_bit_indices=one_bit.to(device=key_states.device),
            two_bit_indices=two_bit.to(device=key_states.device),
            one_bit_channels=one_count,
            two_bit_channels=two_count,
        )


def _validate_dtqi_inputs(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    num_key_value_groups: int,
    recent_query_window: int,
) -> None:
    if not isinstance(query_states, torch.Tensor) or query_states.ndim != 4:
        raise ValueError("query_states must have shape [B, H_q, T, D]")
    if not isinstance(key_states, torch.Tensor) or key_states.ndim != 4:
        raise ValueError("key_states must have shape [B, H_kv, T, D]")
    if not query_states.is_floating_point() or not key_states.is_floating_point():
        raise ValueError("CAGE-v4-DTQI inputs must be floating point")
    if not bool(torch.isfinite(query_states).all()) or not bool(torch.isfinite(key_states).all()):
        raise ValueError("CAGE-v4-DTQI inputs must be finite")
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[2:] != key_states.shape[2:]:
        raise ValueError("Query and Key batch, sequence, and channel shapes must match")
    if isinstance(num_key_value_groups, bool) or not isinstance(num_key_value_groups, int):
        raise ValueError("num_key_value_groups must be a positive integer")
    if num_key_value_groups <= 0:
        raise ValueError("num_key_value_groups must be a positive integer")
    if query_states.shape[1] != key_states.shape[1] * num_key_value_groups:
        raise ValueError("num_key_value_groups does not match Query and KV head counts")
    if query_states.shape[2] <= 0:
        raise ValueError("CAGE-v4-DTQI requires at least one prompt token")
    if isinstance(recent_query_window, bool) or not isinstance(recent_query_window, int):
        raise ValueError("recent_query_window must be a positive integer")
    if recent_query_window <= 0:
        raise ValueError("recent_query_window must be a positive integer")


__all__ = [
    "CageV4DTQIConfig",
    "DTQI_GLOBAL_QUERY_WEIGHT",
    "DTQI_RECENT_QUERY_WEIGHT",
    "Qwen3CageV4DTQICache",
    "Qwen3CageV4DTQILayerPolicy",
    "compute_dtqi_key_importance",
]
