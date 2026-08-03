from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional

import torch

from models.cage_config import CageConfig
from models.cage_importance import (
    assign_channel_buckets,
    compute_key_importance,
    compute_value_importance,
    fixed_random_importance_like,
)
from models.cage_quant import (
    fake_quant_k_by_channel_buckets,
    fake_quant_v_by_channel_buckets,
)

try:
    from transformers.cache_utils import DynamicCache
except ImportError as exc:  # pragma: no cover - exercised by the server dependency gate
    raise ImportError("Qwen3 CAGE requires Hugging Face Transformers with Cache support") from exc


@dataclass(frozen=True)
class Qwen3CageLayerPolicy:
    key_bucket_indices: tuple[torch.Tensor, ...]
    value_bucket_indices: tuple[torch.Tensor, ...]
    key_group_sizes: tuple[int, ...]
    value_group_sizes: tuple[int, ...]
    key_clip_percentiles: tuple[float, ...]
    value_clip_percentiles: tuple[float, ...]


class Qwen3CageCache(DynamicCache):
    """Accuracy-simulation cache for CAGE on the Transformers 4.53.2 Qwen3 API.

    Stored K/V tensors are fake-quantized FP tensors. They are not a packed
    runtime representation and must not be used to claim realized CUDA memory.
    """

    def __init__(
        self,
        cage_config: CageConfig,
        *,
        residual_length: int,
        bits: int = 2,
        require_batch_size_one: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(cage_config, CageConfig):
            raise TypeError("cage_config must be a CageConfig")
        if not cage_config.cage_enable:
            raise ValueError("Qwen3CageCache requires cage_enable=True")
        if cage_config.cage_mode != "fake":
            raise ValueError("Qwen3CageCache supports only cage_mode='fake'")
        if isinstance(residual_length, bool) or not isinstance(residual_length, int) or residual_length <= 0:
            raise ValueError("residual_length must be a positive integer")
        if bits != 2:
            raise ValueError("the frozen Qwen3 CAGE protocol requires bits=2")
        if not isinstance(require_batch_size_one, bool):
            raise TypeError("require_batch_size_one must be a bool")

        self.cage_config = cage_config
        self.residual_length = residual_length
        self.bits = bits
        self.require_batch_size_one = require_batch_size_one
        self.layer_policies: list[Qwen3CageLayerPolicy] = []
        self.key_quantized_lengths: list[int] = []
        self.value_quantized_lengths: list[int] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_update_inputs(key_states, value_states, layer_idx, cache_kwargs)
        if len(self.key_cache) < layer_idx:
            raise ValueError("Qwen3CageCache does not support skipped decoder layers")

        is_prefill = len(self.key_cache) == layer_idx
        if not is_prefill and key_states.shape[-2] != 1:
            raise ValueError(
                "Qwen3 CAGE continuation supports one token per update, "
                f"got {key_states.shape[-2]}"
            )
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if is_prefill:
            policy = self._build_layer_policy(
                query_states=cache_kwargs["query_states"],
                key_states=key_states,
                value_states=value_states,
                o_proj_weight=cache_kwargs["o_proj_weight"],
                layer_idx=layer_idx,
            )
            keys_to_return = key_states
            values_to_return = value_states
            key_storage, key_quantized_length = self._quantize_new_key_prefix(
                key_states.detach().clone(),
                old_quantized_length=0,
                policy=policy,
            )
            value_storage, value_quantized_length = self._quantize_new_value_prefix(
                value_states.detach().clone(),
                old_quantized_length=0,
                policy=policy,
            )
            self.key_cache.append(key_storage)
            self.value_cache.append(value_storage)
            self.layer_policies.append(policy)
            self.key_quantized_lengths.append(key_quantized_length)
            self.value_quantized_lengths.append(value_quantized_length)
            return keys_to_return, values_to_return

        policy = self.layer_policies[layer_idx]
        keys_to_return = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
        values_to_return = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        key_storage, key_quantized_length = self._quantize_new_key_prefix(
            keys_to_return.detach().clone(),
            old_quantized_length=self.key_quantized_lengths[layer_idx],
            policy=policy,
        )
        value_storage, value_quantized_length = self._quantize_new_value_prefix(
            values_to_return.detach().clone(),
            old_quantized_length=self.value_quantized_lengths[layer_idx],
            policy=policy,
        )
        self.key_cache[layer_idx] = key_storage
        self.value_cache[layer_idx] = value_storage
        self.key_quantized_lengths[layer_idx] = key_quantized_length
        self.value_quantized_lengths[layer_idx] = value_quantized_length
        return keys_to_return, values_to_return

    def crop(self, max_length: int):
        if max_length < self.get_seq_length():
            raise NotImplementedError("Qwen3CageCache does not support cache cropping")

    def batch_split(self, full_batch_size: int, split_size: int):
        raise NotImplementedError("Qwen3CageCache does not support generation cache splitting")

    def _validate_update_inputs(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]],
    ) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("Qwen3 CAGE K/V states must have shape [B, H_kv, T, D]")
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Qwen3 CAGE K/V shapes must match, got {tuple(key_states.shape)} and {tuple(value_states.shape)}"
            )
        if self.require_batch_size_one and key_states.shape[0] != 1:
            raise ValueError("the frozen Qwen3 CAGE protocol requires batch_size=1")
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")
        if cache_kwargs is None:
            raise ValueError("Qwen3 CAGE cache update requires cache_kwargs")
        for name in ("query_states", "o_proj_weight"):
            if name not in cache_kwargs:
                raise ValueError(f"Qwen3 CAGE cache update requires cache_kwargs[{name!r}]")
        query_states = cache_kwargs["query_states"]
        if query_states.ndim != 4:
            raise ValueError("query_states must have shape [B, H_q, T, D]")
        if query_states.shape[0] != key_states.shape[0] or query_states.shape[2:] != key_states.shape[2:]:
            raise ValueError("query_states must match K/V batch, sequence, and head dimensions")
        if query_states.shape[1] % key_states.shape[1] != 0:
            raise ValueError("the number of Query heads must be divisible by the number of KV heads")

    def _build_layer_policy(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        o_proj_weight: torch.Tensor,
        layer_idx: int,
    ) -> Qwen3CageLayerPolicy:
        num_query_heads = query_states.shape[1]
        num_key_value_heads = key_states.shape[1]
        head_dim = key_states.shape[-1]
        num_key_value_groups = num_query_heads // num_key_value_heads

        key_bucket_indices: tuple[torch.Tensor, ...] = ()
        value_bucket_indices: tuple[torch.Tensor, ...] = ()
        if self.cage_config.cage_k_enable:
            key_importance = compute_key_importance(
                query_states,
                key_states,
                num_key_value_groups=num_key_value_groups,
            )
            key_scores = self._assignment_scores(
                key_importance,
                policy=self.cage_config.cage_k_importance,
                seed=self.cage_config.cage_assignment_seed + 2 * layer_idx,
            )
            key_bucket_indices = tuple(
                index.to(device=key_states.device)
                for index in assign_channel_buckets(
                    key_scores,
                    num_buckets=self.cage_config.cage_k_num_buckets,
                ).bucket_indices
            )

        if self.cage_config.cage_v_enable:
            value_importance = compute_value_importance(
                value_states,
                o_proj_weight,
                num_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
            value_scores = self._assignment_scores(
                value_importance,
                policy=self.cage_config.cage_v_importance,
                seed=self.cage_config.cage_assignment_seed + 2 * layer_idx + 1,
            )
            value_bucket_indices = tuple(
                index.to(device=value_states.device)
                for index in assign_channel_buckets(
                    value_scores,
                    num_buckets=self.cage_config.cage_v_num_buckets,
                ).bucket_indices
            )

        return Qwen3CageLayerPolicy(
            key_bucket_indices=key_bucket_indices,
            value_bucket_indices=value_bucket_indices,
            key_group_sizes=tuple(self.cage_config.cage_k_group_sizes[: len(key_bucket_indices)]),
            value_group_sizes=tuple(self.cage_config.cage_v_group_sizes[: len(value_bucket_indices)]),
            key_clip_percentiles=tuple(
                self.cage_config.cage_k_clip_percentiles[: len(key_bucket_indices)]
            ),
            value_clip_percentiles=tuple(
                self.cage_config.cage_v_clip_percentiles[: len(value_bucket_indices)]
            ),
        )

    def _assignment_scores(
        self,
        importance: torch.Tensor,
        *,
        policy: str,
        seed: int,
    ) -> torch.Tensor:
        if policy in {"q2_var", "wo_var"}:
            return importance
        if policy == "fixed_random" and self.cage_config.cage_ablation:
            return fixed_random_importance_like(importance, seed=seed)
        raise ValueError(
            f"unsupported Qwen3 CAGE assignment policy {policy!r} for "
            f"cage_ablation={self.cage_config.cage_ablation}"
        )

    def _quantize_new_key_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
        policy: Qwen3CageLayerPolicy,
    ) -> tuple[torch.Tensor, int]:
        if not self.cage_config.cage_k_enable:
            return states, 0
        new_quantized_length = states.shape[-2] - states.shape[-2] % self.residual_length
        if new_quantized_length > old_quantized_length:
            states[:, :, old_quantized_length:new_quantized_length, :] = fake_quant_k_by_channel_buckets(
                states[:, :, old_quantized_length:new_quantized_length, :].contiguous(),
                bucket_indices=policy.key_bucket_indices,
                group_sizes=policy.key_group_sizes,
                clip_percentiles=policy.key_clip_percentiles,
                bits=self.bits,
            )
        return states, new_quantized_length

    def _quantize_new_value_prefix(
        self,
        states: torch.Tensor,
        *,
        old_quantized_length: int,
        policy: Qwen3CageLayerPolicy,
    ) -> tuple[torch.Tensor, int]:
        if not self.cage_config.cage_v_enable:
            return states, 0
        new_quantized_length = max(0, states.shape[-2] - self.residual_length)
        if new_quantized_length > old_quantized_length:
            states[:, :, old_quantized_length:new_quantized_length, :] = fake_quant_v_by_channel_buckets(
                states[:, :, old_quantized_length:new_quantized_length, :].contiguous(),
                bucket_indices=policy.value_bucket_indices,
                group_sizes=policy.value_group_sizes,
                clip_percentiles=policy.value_clip_percentiles,
                bits=self.bits,
            )
        return states, new_quantized_length


def install_qwen3_cage_attention(model: torch.nn.Module) -> int:
    """Install the CAGE Cache adapter on standard Qwen3 attention instances."""

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    except ImportError as exc:
        raise ImportError(
            "Qwen3 CAGE requires the frozen Transformers 4.53.2 Qwen3 implementation"
        ) from exc

    installed = 0
    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue
        if hasattr(module, "_qwen3_cage_original_forward"):
            continue
        module._qwen3_cage_original_forward = module.forward
        module.forward = MethodType(_qwen3_cage_attention_forward, module)
        installed += 1
    if installed == 0 and not any(
        hasattr(module, "_qwen3_cage_original_forward") for module in model.modules()
    ):
        raise ValueError("no standard Qwen3Attention modules were found")
    return installed


def uninstall_qwen3_cage_attention(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        if not hasattr(module, "_qwen3_cage_original_forward"):
            continue
        module.forward = module._qwen3_cage_original_forward
        delattr(module, "_qwen3_cage_original_forward")
        restored += 1
    return restored


def _qwen3_cage_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[DynamicCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from transformers.models.qwen3 import modeling_qwen3

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    if past_key_value is not None:
        cache_kwargs: dict[str, Any] = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        if isinstance(past_key_value, Qwen3CageCache):
            cache_kwargs.update(
                query_states=query_states,
                o_proj_weight=self.o_proj.weight,
            )
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    attention_interface = modeling_qwen3.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


__all__ = [
    "Qwen3CageCache",
    "Qwen3CageLayerPolicy",
    "install_qwen3_cage_attention",
    "uninstall_qwen3_cage_attention",
]
