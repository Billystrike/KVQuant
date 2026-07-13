from __future__ import annotations

from collections.abc import Sequence

import torch

from utils.cage_experiment_schema import ExperimentPointError


_SCALE_EPS = 1e-8


def fake_quant_k_by_channel_buckets(
    key_states: torch.Tensor,
    bucket_indices: Sequence[torch.Tensor],
    group_sizes: Sequence[int],
    clip_percentiles: Sequence[float | None],
    bits: int = 2,
) -> torch.Tensor:
    """Fake-quantize selected key channels bucket-by-bucket.

    key_states is shaped [B, H_kv, T, D]. Each bucket selects channels from D
    and quantizes those selected channels along the token dimension T.
    """
    _require_4d("key_states", key_states)
    return _fake_quant_by_channel_buckets(
        states=key_states,
        bucket_indices=bucket_indices,
        group_sizes=group_sizes,
        clip_percentiles=clip_percentiles,
        bits=bits,
        quantize_dim=2,
    )


def fake_quant_v_by_channel_buckets(
    value_states: torch.Tensor,
    bucket_indices: Sequence[torch.Tensor],
    group_sizes: Sequence[int],
    clip_percentiles: Sequence[float | None],
    bits: int = 2,
) -> torch.Tensor:
    """Fake-quantize selected value channels bucket-by-bucket.

    value_states is shaped [B, H_kv, T, D]. Each bucket selects channels from D
    and quantizes those selected channels along the selected channel dimension.
    """
    _require_4d("value_states", value_states)
    return _fake_quant_by_channel_buckets(
        states=value_states,
        bucket_indices=bucket_indices,
        group_sizes=group_sizes,
        clip_percentiles=clip_percentiles,
        bits=bits,
        quantize_dim=3,
    )


def _fake_quant_by_channel_buckets(
    states: torch.Tensor,
    bucket_indices: Sequence[torch.Tensor],
    group_sizes: Sequence[int],
    clip_percentiles: Sequence[float | None],
    bits: int,
    quantize_dim: int,
) -> torch.Tensor:
    _require_floating_tensor("states", states)
    _require_finite_tensor("states", states)
    _require_positive_int("bits", bits)

    buckets = tuple(bucket_indices)
    _require_bucket_parameters(buckets, group_sizes, clip_percentiles)

    batch_size, num_heads, sequence_length, head_dim = states.shape
    output = states.clone()
    for bucket_id, raw_indices in enumerate(buckets):
        indices = _normalize_bucket_indices(
            raw_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            device=states.device,
        )
        if indices.shape[1] == 0:
            continue

        gather_index = indices.view(1, num_heads, 1, -1).expand(
            batch_size,
            num_heads,
            sequence_length,
            indices.shape[1],
        )
        selected = states.gather(dim=-1, index=gather_index)
        quantized = _asymmetric_fake_quant_grouped(
            selected,
            quantize_dim=quantize_dim,
            group_size=group_sizes[bucket_id],
            clip_percentile=clip_percentiles[bucket_id],
            bits=bits,
        )
        output.scatter_(dim=-1, index=gather_index, src=quantized)

    _require_finite_tensor("fake-quantized states", output)
    return output


def _asymmetric_fake_quant_grouped(
    tensor: torch.Tensor,
    quantize_dim: int,
    group_size: int,
    clip_percentile: float | None,
    bits: int,
) -> torch.Tensor:
    _require_positive_int("group_size", group_size)
    _require_clip_percentile(clip_percentile)

    quant_dim = quantize_dim if quantize_dim >= 0 else tensor.ndim + quantize_dim
    if quant_dim < 0 or quant_dim >= tensor.ndim:
        raise ValueError(f"quantize_dim={quantize_dim} is out of range for shape {tuple(tensor.shape)}")

    original_dtype = tensor.dtype
    moved = tensor.float().movedim(quant_dim, -1).contiguous()
    original_shape = moved.shape
    quant_length = original_shape[-1]
    padded = _pad_last_dim_by_repeating_last(moved, group_size)
    padded_length = padded.shape[-1]

    grouped = padded.reshape(*original_shape[:-1], padded_length // group_size, group_size)
    mn, mx = _quant_bounds(grouped, clip_percentile)
    clipped = torch.minimum(torch.maximum(grouped, mn), mx)
    scale = (mx - mn).clamp_min(_SCALE_EPS) / (2**bits - 1)
    code = torch.round(((clipped - mn) / scale).clamp(0, 2**bits - 1))
    dequantized = code * scale + mn
    dequantized = dequantized.reshape(*original_shape[:-1], padded_length)[..., :quant_length]
    return dequantized.movedim(-1, quant_dim).to(dtype=original_dtype)


def _quant_bounds(tensor: torch.Tensor, clip_percentile: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    if clip_percentile is None or clip_percentile >= 1.0:
        return tensor.amin(dim=-1, keepdim=True), tensor.amax(dim=-1, keepdim=True)

    lower_q = (1.0 - clip_percentile) / 2.0
    upper_q = 1.0 - lower_q
    return (
        torch.quantile(tensor, lower_q, dim=-1, keepdim=True),
        torch.quantile(tensor, upper_q, dim=-1, keepdim=True),
    )


def _pad_last_dim_by_repeating_last(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    quant_length = tensor.shape[-1]
    pad_count = (group_size - quant_length % group_size) % group_size
    if pad_count == 0:
        return tensor

    pad_values = tensor[..., -1:].expand(*tensor.shape[:-1], pad_count)
    return torch.cat((tensor, pad_values), dim=-1)


def _normalize_bucket_indices(
    bucket_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.as_tensor(bucket_indices, dtype=torch.long, device=device)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(num_heads, -1)
    elif indices.ndim != 2:
        raise ValueError(
            "bucket_indices entries must have shape [D_bucket] or [H, D_bucket], "
            f"got {tuple(indices.shape)}"
        )

    if indices.shape[0] != num_heads:
        raise ValueError(
            f"bucket_indices head dimension must be {num_heads}, got {indices.shape[0]}"
        )
    if indices.shape[1] == 0:
        return indices
    if indices.min().item() < 0 or indices.max().item() >= head_dim:
        raise ValueError(
            f"bucket_indices values must be in [0, {head_dim}), got min={indices.min().item()} "
            f"and max={indices.max().item()}"
        )
    for head_id, head_indices in enumerate(indices):
        if torch.unique(head_indices).numel() != head_indices.numel():
            raise ValueError(f"bucket_indices for head {head_id} contains duplicate channels")
    return indices


def _require_bucket_parameters(
    bucket_indices: tuple[torch.Tensor, ...],
    group_sizes: Sequence[int],
    clip_percentiles: Sequence[float | None],
) -> None:
    if len(bucket_indices) == 0:
        raise ValueError("bucket_indices must contain at least one bucket")
    if len(group_sizes) < len(bucket_indices):
        raise ValueError(
            f"group_sizes must have at least {len(bucket_indices)} entries, got {len(group_sizes)}"
        )
    if len(clip_percentiles) < len(bucket_indices):
        raise ValueError(
            "clip_percentiles must have at least "
            f"{len(bucket_indices)} entries, got {len(clip_percentiles)}"
        )
    for bucket_id in range(len(bucket_indices)):
        _require_positive_int(f"group_sizes[{bucket_id}]", group_sizes[bucket_id])
        _require_clip_percentile(clip_percentiles[bucket_id])


def _require_4d(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, H_kv, T, D], got {tuple(tensor.shape)}")


def _require_floating_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor, got {tensor.dtype}")


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ExperimentPointError(f"{name} contains non-finite values")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_clip_percentile(value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (float, int)) or not 0.0 < float(value) <= 1.0:
        raise ValueError(f"clip_percentile must be in (0, 1] or None, got {value!r}")


__all__ = [
    "fake_quant_k_by_channel_buckets",
    "fake_quant_v_by_channel_buckets",
]
