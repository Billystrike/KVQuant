from __future__ import annotations

import torch

from models.cage_quant import fake_quant_k_uniform
from utils.cage_experiment_schema import ExperimentPointError


_SCALE_EPS = 1e-8


def fake_quant_k_sparse_refinement(
    key_states: torch.Tensor,
    *,
    one_bit_indices: torch.Tensor,
    two_bit_indices: torch.Tensor,
    base_group_size: int = 128,
    refinement_group_size: int = 128,
    base_clip_percentile: float | None = 1.0,
    refinement_clip_percentile: float | None = 1.0,
) -> torch.Tensor:
    """Apply an INT2 Key backbone plus disjoint sparse residual refinements.

    Selected channels receive a separately quantized residual code with either
    one or two additional bits.  Returned tensors remain floating point; this
    function is an accuracy simulation rather than packed runtime storage.
    """

    _require_key_tensor(key_states)
    _require_positive_int("base_group_size", base_group_size)
    _require_positive_int("refinement_group_size", refinement_group_size)
    _require_clip_percentile(base_clip_percentile)
    _require_clip_percentile(refinement_clip_percentile)

    _, num_heads, _, head_dim = key_states.shape
    one_bit = _normalize_indices(
        one_bit_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        device=key_states.device,
    )
    two_bit = _normalize_indices(
        two_bit_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        device=key_states.device,
    )
    for head_id in range(num_heads):
        if one_bit.shape[1] and two_bit.shape[1]:
            overlap = torch.isin(one_bit[head_id], two_bit[head_id])
            if bool(overlap.any()):
                raise ValueError(
                    f"one_bit_indices and two_bit_indices overlap for head {head_id}"
                )

    base = _asymmetric_fake_quant_grouped(
        key_states,
        quantize_dim=2,
        group_size=base_group_size,
        clip_percentile=base_clip_percentile,
        bits=2,
    )
    residual = key_states - base
    output = base.clone()
    for indices, bits in ((one_bit, 1), (two_bit, 2)):
        if indices.shape[1] == 0:
            continue
        gather_index = indices.view(1, num_heads, 1, -1).expand(
            key_states.shape[0],
            num_heads,
            key_states.shape[2],
            indices.shape[1],
        )
        selected_residual = residual.gather(dim=-1, index=gather_index)
        residual_hat = _asymmetric_fake_quant_grouped(
            selected_residual,
            quantize_dim=2,
            group_size=refinement_group_size,
            clip_percentile=refinement_clip_percentile,
            bits=bits,
        )
        refined = base.gather(dim=-1, index=gather_index) + residual_hat
        output.scatter_(dim=-1, index=gather_index, src=refined)

    _require_finite("sparse-refined Key states", output)
    return output


def _asymmetric_fake_quant_grouped(
    tensor: torch.Tensor,
    *,
    quantize_dim: int,
    group_size: int,
    clip_percentile: float | None,
    bits: int,
) -> torch.Tensor:
    original_dtype = tensor.dtype
    moved = tensor.float().movedim(quantize_dim, -1).contiguous()
    original_shape = moved.shape
    quant_length = original_shape[-1]
    pad_count = (group_size - quant_length % group_size) % group_size
    if pad_count:
        moved = torch.cat(
            (moved, moved[..., -1:].expand(*moved.shape[:-1], pad_count)),
            dim=-1,
        )
    grouped = moved.reshape(*original_shape[:-1], moved.shape[-1] // group_size, group_size)
    if clip_percentile is None or clip_percentile >= 1.0:
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
    else:
        lower = (1.0 - clip_percentile) / 2.0
        mn = torch.quantile(grouped, lower, dim=-1, keepdim=True)
        mx = torch.quantile(grouped, 1.0 - lower, dim=-1, keepdim=True)
    clipped = torch.minimum(torch.maximum(grouped, mn), mx)
    scale = (mx - mn).clamp_min(_SCALE_EPS) / (2**bits - 1)
    code = torch.round(((clipped - mn) / scale).clamp(0, 2**bits - 1))
    dequantized = (code * scale + mn).reshape(*original_shape[:-1], moved.shape[-1])
    return dequantized[..., :quant_length].movedim(-1, quantize_dim).to(original_dtype)


def _normalize_indices(
    raw: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.as_tensor(raw, dtype=torch.long, device=device)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(num_heads, -1)
    elif indices.ndim != 2:
        raise ValueError("refinement indices must have shape [D_selected] or [H, D_selected]")
    if indices.shape[0] != num_heads:
        raise ValueError("refinement indices head dimension mismatch")
    if indices.shape[1] == 0:
        return indices
    if indices.min().item() < 0 or indices.max().item() >= head_dim:
        raise ValueError("refinement index is outside the Key channel dimension")
    for head_id, head_indices in enumerate(indices):
        if torch.unique(head_indices).numel() != head_indices.numel():
            raise ValueError(f"refinement indices contain duplicates for head {head_id}")
    return indices


def _require_key_tensor(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError("key_states must have shape [B, H_kv, T, D]")
    if not tensor.is_floating_point():
        raise ValueError("key_states must be floating point")
    _require_finite("key_states", tensor)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ExperimentPointError(f"{name} contains non-finite values")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_clip_percentile(value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (float, int)) or not 0.0 < float(value) <= 1.0:
        raise ValueError("clip percentile must be in (0, 1] or None")


__all__ = ["fake_quant_k_sparse_refinement", "fake_quant_k_uniform"]
