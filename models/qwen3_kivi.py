from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from models.cage_quant import fake_quant_k_uniform, fake_quant_v_uniform

try:
    from transformers.cache_utils import DynamicCache
except ImportError as exc:  # pragma: no cover - exercised by the server dependency gate
    raise ImportError("Qwen3 KIVI requires Hugging Face Transformers with Cache support") from exc


@dataclass(frozen=True)
class Qwen3KiviCacheConfig:
    group_size: int
    residual_length: int
    bits: int = 2
    require_batch_size_one: bool = True

    def __post_init__(self) -> None:
        for name in ("group_size", "residual_length", "bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bits != 2:
            raise ValueError("the frozen Qwen3 KIVI protocol requires bits=2")
        if self.residual_length % self.group_size != 0:
            raise ValueError("residual_length must be a multiple of group_size")
        if not isinstance(self.require_batch_size_one, bool):
            raise TypeError("require_batch_size_one must be a bool")


class Qwen3KiviCache(DynamicCache):
    """KIVI-style accuracy-simulation cache on the Qwen3 Cache API.

    Key is quantized per channel along tokens and Value per token along the
    channel dimension. Persistent tensors are dequantized FP values; packed
    bytes are supplied only by the frozen paper estimator.
    """

    def __init__(self, cache_config: Qwen3KiviCacheConfig) -> None:
        super().__init__()
        if not isinstance(cache_config, Qwen3KiviCacheConfig):
            raise TypeError("cache_config must be a Qwen3KiviCacheConfig")
        self.cache_config = cache_config
        self.group_size = cache_config.group_size
        self.residual_length = cache_config.residual_length
        self.bits = cache_config.bits
        self.require_batch_size_one = cache_config.require_batch_size_one
        self.key_quantized_lengths: list[int] = []
        self.value_quantized_lengths: list[int] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self._validate_update_inputs(key_states, value_states, layer_idx)
        if len(self.key_cache) < layer_idx:
            raise ValueError("Qwen3KiviCache does not support skipped decoder layers")

        is_prefill = len(self.key_cache) == layer_idx
        if not is_prefill and key_states.shape[-2] != 1:
            raise ValueError(
                "Qwen3 KIVI continuation supports one token per update, "
                f"got {key_states.shape[-2]}"
            )
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if is_prefill:
            keys_to_return = key_states
            values_to_return = value_states
            key_storage, key_quantized_length = self._quantize_new_key_prefix(
                key_states.detach().clone(),
                old_quantized_length=0,
            )
            value_storage, value_quantized_length = self._quantize_new_value_prefix(
                value_states.detach().clone(),
                old_quantized_length=0,
            )
            self.key_cache.append(key_storage)
            self.value_cache.append(value_storage)
            self.key_quantized_lengths.append(key_quantized_length)
            self.value_quantized_lengths.append(value_quantized_length)
            return keys_to_return, values_to_return

        keys_to_return = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
        values_to_return = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        key_storage, key_quantized_length = self._quantize_new_key_prefix(
            keys_to_return.detach().clone(),
            old_quantized_length=self.key_quantized_lengths[layer_idx],
        )
        value_storage, value_quantized_length = self._quantize_new_value_prefix(
            values_to_return.detach().clone(),
            old_quantized_length=self.value_quantized_lengths[layer_idx],
        )
        self.key_cache[layer_idx] = key_storage
        self.value_cache[layer_idx] = value_storage
        self.key_quantized_lengths[layer_idx] = key_quantized_length
        self.value_quantized_lengths[layer_idx] = value_quantized_length
        return keys_to_return, values_to_return

    def crop(self, max_length: int):
        if max_length < self.get_seq_length():
            raise NotImplementedError("Qwen3KiviCache does not support cache cropping")

    def batch_split(self, full_batch_size: int, split_size: int):
        raise NotImplementedError("Qwen3KiviCache does not support generation cache splitting")

    def _validate_update_inputs(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("Qwen3 KIVI K/V states must have shape [B, H_kv, T, D]")
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Qwen3 KIVI K/V shapes must match, got {tuple(key_states.shape)} and {tuple(value_states.shape)}"
            )
        if self.require_batch_size_one and key_states.shape[0] != 1:
            raise ValueError("the frozen Qwen3 KIVI protocol requires batch_size=1")
        if key_states.shape[-1] % self.group_size != 0:
            raise ValueError("Qwen3 KIVI head_dim must be divisible by group_size")
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")

    def _quantize_new_key_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
    ) -> tuple[torch.Tensor, int]:
        new_quantized_length = states.shape[-2] - states.shape[-2] % self.residual_length
        if new_quantized_length > old_quantized_length:
            states[:, :, old_quantized_length:new_quantized_length, :] = fake_quant_k_uniform(
                states[:, :, old_quantized_length:new_quantized_length, :].contiguous(),
                group_size=self.group_size,
                bits=self.bits,
            )
        return states, new_quantized_length

    def _quantize_new_value_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
    ) -> tuple[torch.Tensor, int]:
        new_quantized_length = max(0, states.shape[-2] - self.residual_length)
        if new_quantized_length > old_quantized_length:
            states[:, :, old_quantized_length:new_quantized_length, :] = fake_quant_v_uniform(
                states[:, :, old_quantized_length:new_quantized_length, :].contiguous(),
                group_size=self.group_size,
                bits=self.bits,
            )
        return states, new_quantized_length


__all__ = ["Qwen3KiviCache", "Qwen3KiviCacheConfig"]
