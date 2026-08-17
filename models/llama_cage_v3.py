from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from models.cage_importance import compute_key_importance
from models.cage_quant import fake_quant_v_uniform
from models.cage_v2_quant import fake_quant_k_sparse_refinement


LLAMA_CAGE_V3_CACHE_TAG = "LLAMA_CAGE_V3_FAKE_CACHE_V1"


@dataclass(frozen=True)
class LlamaCageV3Plan:
    prompt_length: int
    residual_length: int
    sink_length: int
    layer_two_bit_channel_quotas: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive("prompt_length", self.prompt_length)
        _positive("residual_length", self.residual_length)
        _nonnegative("sink_length", self.sink_length)
        if self.sink_length >= self.prompt_length:
            raise ValueError("sink_length must be smaller than prompt_length")
        if not self.layer_two_bit_channel_quotas:
            raise ValueError("layer quota sequence must not be empty")
        for index, count in enumerate(self.layer_two_bit_channel_quotas):
            _nonnegative(f"layer_two_bit_channel_quotas[{index}]", count)


@dataclass(frozen=True)
class LlamaCageV3Config:
    plans: tuple[LlamaCageV3Plan, ...]
    key_base_group_size: int = 128
    key_refinement_group_size: int = 128
    value_group_size: int = 128

    def __post_init__(self) -> None:
        for name in ("key_base_group_size", "key_refinement_group_size", "value_group_size"):
            _positive(name, getattr(self, name))
        if not self.plans:
            raise ValueError("CAGE-v3 plans must not be empty")
        lengths = [plan.prompt_length for plan in self.plans]
        if len(set(lengths)) != len(lengths):
            raise ValueError("CAGE-v3 prompt lengths must be unique")
        layer_counts = {len(plan.layer_two_bit_channel_quotas) for plan in self.plans}
        if len(layer_counts) != 1:
            raise ValueError("CAGE-v3 plans must have the same layer count")

    def plan_for_prompt(self, prompt_length: int) -> LlamaCageV3Plan:
        matches = [plan for plan in self.plans if plan.prompt_length == prompt_length]
        if len(matches) != 1:
            raise ValueError(f"no unique frozen CAGE-v3 plan for prompt length {prompt_length}")
        return matches[0]

    @property
    def num_hidden_layers(self) -> int:
        return len(self.plans[0].layer_two_bit_channel_quotas)


@dataclass(frozen=True)
class LlamaCageV3LayerCache:
    layer_idx: int
    prompt_length: int
    residual_length: int
    sink_length: int
    two_bit_indices: torch.Tensor
    key_sink: torch.Tensor
    key_quantized: torch.Tensor | None
    key_residual: torch.Tensor | None
    value_sink: torch.Tensor
    value_quantized: torch.Tensor | None
    value_residual: torch.Tensor | None
    kv_seq_len: int

    def __post_init__(self) -> None:
        _nonnegative("layer_idx", self.layer_idx)
        _positive("prompt_length", self.prompt_length)
        _positive("residual_length", self.residual_length)
        _nonnegative("sink_length", self.sink_length)
        _positive("kv_seq_len", self.kv_seq_len)
        if self.two_bit_indices.ndim != 2:
            raise ValueError("two_bit_indices must have shape [H_kv, D_selected]")
        key = reconstruct_key(self)
        value = reconstruct_value(self)
        if key.shape[-2] != self.kv_seq_len or value.shape[-2] != self.kv_seq_len:
            raise ValueError("CAGE-v3 cache tensor lengths do not match kv_seq_len")


def config_from_quota_payload(payload: Mapping[str, Any]) -> LlamaCageV3Config:
    if payload.get("plan_id") != "llama2-7b-cage-v3-depth-normalized-quota-plan-v1":
        raise ValueError("Llama CAGE-v3 quota-plan identity mismatch")
    if payload.get("llama2_metrics_consumed") is not False:
        raise ValueError("Llama CAGE-v3 quota plan must not consume target metrics")
    plans = []
    for raw in payload.get("plans", []):
        plans.append(
            LlamaCageV3Plan(
                prompt_length=int(raw["prompt_length"]),
                residual_length=int(raw["residual_length"]),
                sink_length=int(raw["sink_length"]),
                layer_two_bit_channel_quotas=tuple(raw["layer_two_bit_channel_quotas"]),
            )
        )
    return LlamaCageV3Config(plans=tuple(plans))


def install_llama_cage_v3_config(model_config: Any, quota_payload: Mapping[str, Any]) -> LlamaCageV3Config:
    config = config_from_quota_payload(quota_payload)
    expected = (
        config.num_hidden_layers,
        128,
        32,
        32,
    )
    actual = (
        getattr(model_config, "num_hidden_layers", None),
        getattr(model_config, "hidden_size", 0) // getattr(model_config, "num_attention_heads", 1),
        getattr(model_config, "num_attention_heads", None),
        getattr(model_config, "num_key_value_heads", None),
    )
    if actual != expected:
        raise ValueError(f"Llama CAGE-v3 production architecture mismatch: {actual} != {expected}")
    model_config.cage_v3_enable = True
    model_config.cage_v3_plans = [
        {
            "prompt_length": plan.prompt_length,
            "residual_length": plan.residual_length,
            "sink_length": plan.sink_length,
            "layer_two_bit_channel_quotas": list(plan.layer_two_bit_channel_quotas),
        }
        for plan in config.plans
    ]
    model_config.cage_v3_key_base_group_size = config.key_base_group_size
    model_config.cage_v3_key_refinement_group_size = config.key_refinement_group_size
    model_config.cage_v3_value_group_size = config.value_group_size
    model_config.cage_enable = False
    return config


def config_from_model_config(model_config: Any) -> LlamaCageV3Config | None:
    if getattr(model_config, "cage_v3_enable", False) is not True:
        return None
    raw_plans = getattr(model_config, "cage_v3_plans", None)
    if not isinstance(raw_plans, Sequence) or isinstance(raw_plans, (str, bytes)):
        raise ValueError("cage_v3_plans must be a sequence")
    payload = {
        "plan_id": "llama2-7b-cage-v3-depth-normalized-quota-plan-v1",
        "llama2_metrics_consumed": False,
        "plans": list(raw_plans),
    }
    base = config_from_quota_payload(payload)
    return LlamaCageV3Config(
        plans=base.plans,
        key_base_group_size=int(getattr(model_config, "cage_v3_key_base_group_size", 128)),
        key_refinement_group_size=int(getattr(model_config, "cage_v3_key_refinement_group_size", 128)),
        value_group_size=int(getattr(model_config, "cage_v3_value_group_size", 128)),
    )


def build_prefill_cache(
    config: LlamaCageV3Config,
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
) -> LlamaCageV3LayerCache:
    _validate_states(query_states, key_states, value_states)
    if key_states.shape[0] != 1:
        raise ValueError("Llama CAGE-v3 acceptance/runtime requires batch size one")
    plan = config.plan_for_prompt(int(key_states.shape[-2]))
    if layer_idx < 0 or layer_idx >= len(plan.layer_two_bit_channel_quotas):
        raise ValueError(f"layer_idx {layer_idx} is outside the frozen quota plan")
    quota = int(plan.layer_two_bit_channel_quotas[layer_idx])
    if quota > key_states.shape[-1]:
        raise ValueError("frozen refinement quota exceeds Llama head_dim")
    groups = query_states.shape[1] // key_states.shape[1]
    importance = compute_key_importance(
        query_states,
        key_states,
        num_key_value_groups=groups,
    )
    two_bit_indices = _select_sparse_refinement_channels(
        importance,
        two_bit_channels=quota,
    )
    sink = plan.sink_length
    non_sink = key_states.shape[-2] - sink
    key_quantized_tokens = non_sink - non_sink % plan.residual_length
    value_quantized_tokens = max(0, non_sink - plan.residual_length)
    key_sink = key_states[:, :, :sink, :].contiguous()
    value_sink = value_states[:, :, :sink, :].contiguous()
    key_quantized_source = key_states[:, :, sink : sink + key_quantized_tokens, :].contiguous()
    value_quantized_source = value_states[:, :, sink : sink + value_quantized_tokens, :].contiguous()
    key_residual = key_states[:, :, sink + key_quantized_tokens :, :].contiguous()
    value_residual = value_states[:, :, sink + value_quantized_tokens :, :].contiguous()
    key_quantized = _quantize_key(config, key_quantized_source, two_bit_indices)
    value_quantized = _quantize_value(config, value_quantized_source)
    return LlamaCageV3LayerCache(
        layer_idx=layer_idx,
        prompt_length=plan.prompt_length,
        residual_length=plan.residual_length,
        sink_length=plan.sink_length,
        two_bit_indices=two_bit_indices,
        key_sink=key_sink,
        key_quantized=key_quantized,
        key_residual=_none_if_empty(key_residual),
        value_sink=value_sink,
        value_quantized=value_quantized,
        value_residual=_none_if_empty(value_residual),
        kv_seq_len=int(key_states.shape[-2]),
    )


def append_decode_token(
    config: LlamaCageV3Config,
    cache: LlamaCageV3LayerCache,
    *,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> LlamaCageV3LayerCache:
    if key_states.shape != value_states.shape or key_states.ndim != 4:
        raise ValueError("decode Key/Value states must have the same [B,H,1,D] shape")
    if key_states.shape[0] != 1 or key_states.shape[-2] != 1:
        raise ValueError("Llama CAGE-v3 continuation accepts exactly one token and batch size one")
    if key_states.shape[1] != cache.two_bit_indices.shape[0]:
        raise ValueError("decode KV-head count differs from the frozen cache policy")
    key_residual = _concat(cache.key_residual, key_states)
    key_quantized = cache.key_quantized
    if key_residual.shape[-2] == cache.residual_length:
        block = _quantize_key(config, key_residual, cache.two_bit_indices)
        key_quantized = _concat(key_quantized, block)
        key_residual = None
    elif key_residual.shape[-2] > cache.residual_length:
        raise ValueError("Llama CAGE-v3 Key residual exceeded its frozen flush length")

    value_residual = _concat(cache.value_residual, value_states)
    value_quantized = cache.value_quantized
    overflow = value_residual.shape[-2] - cache.residual_length
    if overflow > 0:
        prefix = value_residual[:, :, :overflow, :].contiguous()
        value_quantized = _concat(value_quantized, _quantize_value(config, prefix))
        value_residual = value_residual[:, :, overflow:, :].contiguous()
    return LlamaCageV3LayerCache(
        layer_idx=cache.layer_idx,
        prompt_length=cache.prompt_length,
        residual_length=cache.residual_length,
        sink_length=cache.sink_length,
        two_bit_indices=cache.two_bit_indices,
        key_sink=cache.key_sink,
        key_quantized=key_quantized,
        key_residual=key_residual,
        value_sink=cache.value_sink,
        value_quantized=value_quantized,
        value_residual=value_residual,
        kv_seq_len=cache.kv_seq_len + 1,
    )


def reconstruct_key(cache: LlamaCageV3LayerCache) -> torch.Tensor:
    return _join(cache.key_sink, cache.key_quantized, cache.key_residual)


def reconstruct_value(cache: LlamaCageV3LayerCache) -> torch.Tensor:
    return _join(cache.value_sink, cache.value_quantized, cache.value_residual)


def pack_cache(cache: LlamaCageV3LayerCache) -> tuple[str, LlamaCageV3LayerCache, int]:
    return (LLAMA_CAGE_V3_CACHE_TAG, cache, cache.kv_seq_len)


def is_llama_cage_v3_cache(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and value[0] == LLAMA_CAGE_V3_CACHE_TAG
        and isinstance(value[1], LlamaCageV3LayerCache)
        and value[2] == value[1].kv_seq_len
    )


def unpack_cache(value: Any) -> LlamaCageV3LayerCache:
    if not is_llama_cage_v3_cache(value):
        raise ValueError("past_key_value is not a Llama CAGE-v3 cache")
    return value[1]


def _quantize_key(
    config: LlamaCageV3Config, states: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor | None:
    if states.shape[-2] == 0:
        return None
    return fake_quant_k_sparse_refinement(
        states,
        one_bit_indices=indices[:, :0],
        two_bit_indices=indices,
        base_group_size=config.key_base_group_size,
        refinement_group_size=config.key_refinement_group_size,
    )


def _quantize_value(config: LlamaCageV3Config, states: torch.Tensor) -> torch.Tensor | None:
    if states.shape[-2] == 0:
        return None
    return fake_quant_v_uniform(states, group_size=config.value_group_size, bits=2)


def _validate_states(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V states must have shape [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("Key and Value states must have identical shapes")
    if query.shape[0] != key.shape[0] or query.shape[-2:] != key.shape[-2:]:
        raise ValueError("Query batch, sequence, and head_dim must match Key")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if not all(tensor.is_floating_point() and bool(torch.isfinite(tensor).all()) for tensor in (query, key, value)):
        raise ValueError("Q/K/V states must be finite floating point tensors")


def _select_sparse_refinement_channels(
    importance: torch.Tensor,
    *,
    two_bit_channels: int,
) -> torch.Tensor:
    """Select the highest-score channels with deterministic channel-index ties.

    This is intentionally local to the Llama runtime.  Importing the equivalent
    Qwen3 helper would couple the frozen Transformers-4.43.1 Llama environment
    to Qwen3-only model modules.
    """

    if not isinstance(importance, torch.Tensor) or importance.ndim != 2:
        raise ValueError("importance must have shape [H_kv, D]")
    if not importance.is_floating_point() or not bool(torch.isfinite(importance).all()):
        raise ValueError("importance must be a finite floating point tensor")
    _nonnegative("two_bit_channels", two_bit_channels)
    if two_bit_channels > importance.shape[1]:
        raise ValueError("refinement channel count exceeds the channel dimension")
    ordered = torch.argsort(importance, dim=-1, descending=True, stable=True)
    return ordered[:, :two_bit_channels]


def _none_if_empty(tensor: torch.Tensor) -> torch.Tensor | None:
    return None if tensor.shape[-2] == 0 else tensor


def _concat(prefix: torch.Tensor | None, suffix: torch.Tensor) -> torch.Tensor:
    return suffix if prefix is None else torch.cat((prefix, suffix), dim=2)


def _join(*parts: torch.Tensor | None) -> torch.Tensor:
    present = [part for part in parts if part is not None]
    if not present:
        raise ValueError("CAGE-v3 cache side is empty")
    return torch.cat(present, dim=2) if len(present) > 1 else present[0]


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


__all__ = [
    "LlamaCageV3Config",
    "LlamaCageV3LayerCache",
    "LlamaCageV3Plan",
    "append_decode_token",
    "build_prefill_cache",
    "config_from_model_config",
    "config_from_quota_payload",
    "install_llama_cage_v3_config",
    "is_llama_cage_v3_cache",
    "pack_cache",
    "reconstruct_key",
    "reconstruct_value",
    "unpack_cache",
]
