from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.cage_experiment_schema import ExperimentPointError


@dataclass(frozen=True)
class CageBucketAssignment:
    bucket_indices: tuple[torch.Tensor, ...]
    bucket_rank_map: torch.Tensor
    num_buckets: int
    valid_channel_counts: tuple[int, ...] | None = None


def compute_key_importance(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    num_key_value_groups: int = 1,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """Compute CAGE-KV key channel importance.

    Args:
        query_states: Tensor shaped [B, H_q, T, D].
        key_states: Tensor shaped [B, H_kv, T, D].
        num_key_value_groups: Number of query heads sharing each KV head.
        reduce_batch: If True, average per-batch importance to [H_kv, D].
    """
    _require_4d("query_states", query_states)
    _require_4d("key_states", key_states)
    _require_finite_tensor("query_states", query_states)
    _require_finite_tensor("key_states", key_states)
    _require_positive_int("num_key_value_groups", num_key_value_groups)

    batch_size, num_query_heads, sequence_length, head_dim = query_states.shape
    key_batch_size, num_key_value_heads, key_sequence_length, key_head_dim = key_states.shape
    _require_matching_shape(
        query_name="query_states",
        key_name="key_states",
        query_shape=(batch_size, sequence_length, head_dim),
        key_shape=(key_batch_size, key_sequence_length, key_head_dim),
    )

    num_key_value_groups = _resolve_num_key_value_groups(
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        num_key_value_groups=num_key_value_groups,
    )

    query_energy = query_states.square().mean(dim=2)
    if num_key_value_groups > 1:
        query_energy = query_energy.view(
            batch_size,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        ).sum(dim=2)

    key_variance = key_states.var(dim=2, unbiased=False)
    importance = query_energy * key_variance
    return _finalize_importance(importance, reduce_batch=reduce_batch)


def compute_value_importance(
    value_states: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attn_weights: torch.Tensor | None = None,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """Compute CAGE-KV value channel importance with the MVP no-attention fallback."""
    del attn_weights

    _require_4d("value_states", value_states)
    _require_2d("o_proj_weight", o_proj_weight)
    _require_finite_tensor("value_states", value_states)
    _require_finite_tensor("o_proj_weight", o_proj_weight)
    _require_positive_int("num_heads", num_heads)
    _require_positive_int("num_key_value_heads", num_key_value_heads)
    _require_positive_int("head_dim", head_dim)

    _, value_num_key_value_heads, _, value_head_dim = value_states.shape
    if value_num_key_value_heads != num_key_value_heads:
        raise ValueError(
            "value_states head dimension must match num_key_value_heads, "
            f"got {value_num_key_value_heads} and {num_key_value_heads}"
        )
    if value_head_dim != head_dim:
        raise ValueError(
            f"value_states head_dim must match head_dim, got {value_head_dim} and {head_dim}"
        )
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_key_value_heads={num_key_value_heads}"
        )

    expected_o_proj_columns = num_heads * head_dim
    if o_proj_weight.shape[1] != expected_o_proj_columns:
        raise ValueError(
            "o_proj_weight must have num_heads * head_dim columns, "
            f"got {o_proj_weight.shape[1]} and expected {expected_o_proj_columns}"
        )

    value_variance = value_states.var(dim=2, unbiased=False)
    output_projection_norm = _group_output_projection_norm(
        o_proj_weight=o_proj_weight,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    importance = value_variance * output_projection_norm.unsqueeze(0)
    return _finalize_importance(importance, reduce_batch=reduce_batch)


def assign_channel_buckets(
    importance: torch.Tensor,
    num_buckets: int = 3,
    stable: bool = True,
    pad_to_multiple: int | None = None,
) -> CageBucketAssignment:
    """Assign channels to high-to-low importance buckets for each KV head.

    Args:
        importance: Tensor shaped [H, D] or [B, H, D].
        num_buckets: Requested number of buckets. The effective count is capped at D.
        stable: Kept for API compatibility. Ties are always channel-index stable.
        pad_to_multiple: If set, pad each bucket's channel dimension to this multiple.
    """
    importance = _prepare_bucket_importance(importance)
    _require_positive_int("num_buckets", num_buckets)
    if not isinstance(stable, bool):
        raise ValueError(f"stable must be a bool, got {stable!r}")
    if pad_to_multiple is not None:
        _require_positive_int("pad_to_multiple", pad_to_multiple)

    num_heads, head_dim = importance.shape
    effective_num_buckets = min(num_buckets, head_dim)
    bucket_sizes = _bucket_sizes(head_dim, effective_num_buckets)
    del stable
    sorted_indices = torch.argsort(importance, dim=-1, descending=True, stable=True)

    bucket_rank_map = torch.empty(
        (num_heads, head_dim),
        dtype=torch.long,
        device=importance.device,
    )
    bucket_indices = []
    valid_channel_counts = []
    start = 0
    for bucket_id, bucket_size in enumerate(bucket_sizes):
        end = start + bucket_size
        indices = sorted_indices[:, start:end]
        bucket_rank_map.scatter_(
            dim=1,
            index=indices,
            src=torch.full_like(indices, bucket_id, dtype=torch.long),
        )
        valid_channel_counts.append(bucket_size)
        bucket_indices.append(_pad_bucket_indices(indices, pad_to_multiple))
        start = end

    return CageBucketAssignment(
        bucket_indices=tuple(bucket_indices),
        bucket_rank_map=bucket_rank_map,
        num_buckets=effective_num_buckets,
        valid_channel_counts=tuple(valid_channel_counts) if pad_to_multiple is not None else None,
    )


def _group_output_projection_norm(
    o_proj_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    num_key_value_groups = num_heads // num_key_value_heads
    column_norm = o_proj_weight.square().sum(dim=0)
    column_norm = column_norm.view(num_heads, head_dim)
    if num_key_value_groups == 1:
        return column_norm
    return column_norm.view(num_key_value_heads, num_key_value_groups, head_dim).sum(dim=1)


def _prepare_bucket_importance(importance: torch.Tensor) -> torch.Tensor:
    if not isinstance(importance, torch.Tensor):
        raise TypeError(f"importance must be a torch.Tensor, got {type(importance).__name__}")
    if importance.ndim == 3:
        if importance.shape[0] <= 0:
            raise ValueError(f"importance batch dimension must be non-empty, got {tuple(importance.shape)}")
        importance = importance.mean(dim=0)
    elif importance.ndim != 2:
        raise ValueError(f"importance must have shape [H, D] or [B, H, D], got {tuple(importance.shape)}")

    if importance.shape[0] <= 0 or importance.shape[1] <= 0:
        raise ValueError(f"importance must have non-empty head and channel dimensions, got {tuple(importance.shape)}")
    _require_finite_tensor("importance", importance)
    return importance


def _bucket_sizes(head_dim: int, num_buckets: int) -> tuple[int, ...]:
    base_size = head_dim // num_buckets
    remainder = head_dim % num_buckets
    small_bucket_count = num_buckets - remainder
    sizes = [base_size] * small_bucket_count
    sizes.extend([base_size + 1] * remainder)
    return tuple(sizes)


def _pad_bucket_indices(indices: torch.Tensor, pad_to_multiple: int | None) -> torch.Tensor:
    if pad_to_multiple is None:
        return indices

    valid_count = indices.shape[1]
    padded_count = ((valid_count + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    if padded_count == valid_count:
        return indices

    pad_count = padded_count - valid_count
    pad_values = indices[:, -1:].expand(indices.shape[0], pad_count)
    return torch.cat((indices, pad_values), dim=1)


def _resolve_num_key_value_groups(
    num_query_heads: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
) -> int:
    if num_query_heads == num_key_value_heads and num_key_value_groups == 1:
        return num_key_value_groups

    if num_query_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_key_value_heads={num_key_value_heads}"
        )

    inferred_groups = num_query_heads // num_key_value_heads
    if num_key_value_groups == 1:
        return inferred_groups
    if num_key_value_groups != inferred_groups:
        raise ValueError(
            f"num_key_value_groups={num_key_value_groups} does not match "
            f"num_query_heads / num_key_value_heads = {inferred_groups}"
        )
    return num_key_value_groups


def _finalize_importance(importance: torch.Tensor, reduce_batch: bool) -> torch.Tensor:
    _require_finite_tensor("importance", importance)
    importance = importance.clamp_min(0)
    if reduce_batch:
        importance = importance.mean(dim=0)
    _require_finite_tensor("importance", importance)
    return importance.clamp_min(0)


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ExperimentPointError(f"{name} contains non-finite values")


def _require_4d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, H, T, D], got {tuple(tensor.shape)}")


def _require_2d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [hidden_size, H * D], got {tuple(tensor.shape)}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_matching_shape(
    query_name: str,
    key_name: str,
    query_shape: tuple[int, int, int],
    key_shape: tuple[int, int, int],
) -> None:
    if query_shape != key_shape:
        raise ValueError(
            f"{query_name} and {key_name} must share batch, sequence, and head_dim, "
            f"got {query_shape} and {key_shape}"
        )


__all__ = [
    "CageBucketAssignment",
    "assign_channel_buckets",
    "compute_key_importance",
    "compute_value_importance",
]
