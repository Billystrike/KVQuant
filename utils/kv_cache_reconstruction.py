from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from models.cage_cache import unpack_cage_past_key_value


def reconstruct_kivi_cache(
    past_key_value: Any,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense key/value histories from a KIVI packed cache tuple."""

    from quant.new_pack import unpack_and_dequant_vcache

    k_code, k_full, k_scale, k_min, v_code, v_full, v_scale, v_min, _ = past_key_value
    k_quant = None
    if k_code is not None:
        k_quant = unpack_and_dequant_vcache(
            k_code,
            k_scale.unsqueeze(-1),
            k_min.unsqueeze(-1),
            group_size,
            bits,
        ).transpose(2, 3).contiguous()
    v_quant = None
    if v_code is not None:
        v_quant = unpack_and_dequant_vcache(
            v_code,
            v_scale.unsqueeze(-1),
            v_min.unsqueeze(-1),
            group_size,
            bits,
        )
    return _concat_history(k_quant, k_full), _concat_history(v_quant, v_full)


def reconstruct_cage_cache(past_key_value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense key/value histories from a CAGE fake cache."""

    unpacked = unpack_cage_past_key_value(past_key_value)
    key = _scatter_buckets(
        unpacked.key_cache.key_quant_buckets,
        unpacked.key_cache.key_bucket_indices,
    )
    value = _scatter_buckets(
        unpacked.value_cache.value_quant_buckets,
        unpacked.value_cache.value_bucket_indices,
    )
    return (
        _concat_history(key, unpacked.key_cache.key_full),
        _concat_history(value, unpacked.value_cache.value_full),
    )


def _concat_history(
    quantized: torch.Tensor | None,
    full: torch.Tensor | None,
) -> torch.Tensor:
    if quantized is None:
        if full is None:
            raise ValueError("cache history has neither quantized nor full states")
        return full
    if full is None:
        return quantized
    return torch.cat((quantized, full), dim=2)


def _scatter_buckets(
    quant_buckets: Sequence[torch.Tensor],
    bucket_indices: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    if len(quant_buckets) == 0:
        return None

    first_bucket = quant_buckets[0]
    head_dim = sum(bucket.shape[-1] for bucket in quant_buckets)
    reconstructed = first_bucket.new_zeros(
        first_bucket.shape[0],
        first_bucket.shape[1],
        first_bucket.shape[2],
        head_dim,
    )
    for bucket, indices in zip(quant_buckets, bucket_indices):
        scatter_index = indices.to(device=reconstructed.device).view(
            1,
            indices.shape[0],
            1,
            indices.shape[1],
        ).expand_as(bucket)
        reconstructed.scatter_(
            dim=-1,
            index=scatter_index,
            src=bucket.to(device=reconstructed.device, dtype=reconstructed.dtype),
        )
    return reconstructed


__all__ = ["reconstruct_cage_cache", "reconstruct_kivi_cache"]
