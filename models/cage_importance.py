from __future__ import annotations

import torch


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

    query_energy = torch.nan_to_num(query_states).square().mean(dim=2)
    if num_key_value_groups > 1:
        query_energy = query_energy.view(
            batch_size,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        ).sum(dim=2)

    key_variance = torch.nan_to_num(key_states).var(dim=2, unbiased=False)
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

    value_variance = torch.nan_to_num(value_states).var(dim=2, unbiased=False)
    output_projection_norm = _group_output_projection_norm(
        o_proj_weight=o_proj_weight,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    importance = value_variance * output_projection_norm.unsqueeze(0)
    return _finalize_importance(importance, reduce_batch=reduce_batch)


def _group_output_projection_norm(
    o_proj_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    num_key_value_groups = num_heads // num_key_value_heads
    column_norm = torch.nan_to_num(o_proj_weight).square().sum(dim=0)
    column_norm = column_norm.view(num_heads, head_dim)
    if num_key_value_groups == 1:
        return column_norm
    return column_norm.view(num_key_value_heads, num_key_value_groups, head_dim).sum(dim=1)


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
    importance = torch.nan_to_num(importance).clamp_min(0)
    if reduce_batch:
        importance = importance.mean(dim=0)
    return torch.nan_to_num(importance).clamp_min(0)


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


__all__ = ["compute_key_importance", "compute_value_importance"]
