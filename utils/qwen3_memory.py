from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


ByteReport = dict[str, Any]


def estimate_qwen3_fp16_bytes(
    *,
    seq_len: int,
    batch_size: int = 1,
    num_hidden_layers: int = 36,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bytes_per_value: int = 2,
) -> ByteReport:
    """Return complete active FP16 K/V cache bytes."""

    _require_non_negative("seq_len", seq_len)
    _require_shape(batch_size, num_hidden_layers, num_key_value_heads, head_dim)
    _require_positive("bytes_per_value", bytes_per_value)
    key_bytes = batch_size * num_key_value_heads * seq_len * head_dim * bytes_per_value
    per_layer = {
        "key_fp16_bytes": key_bytes,
        "value_fp16_bytes": key_bytes,
    }
    return _report(
        method="fp16",
        seq_len=seq_len,
        num_hidden_layers=num_hidden_layers,
        token_state={
            "key_quantized_tokens": 0,
            "key_fp16_tokens": seq_len,
            "value_quantized_tokens": 0,
            "value_fp16_tokens": seq_len,
        },
        per_layer=per_layer,
    )


def estimate_qwen3_kivi_bytes(
    *,
    seq_len: int,
    group_size: int,
    residual_length: int,
    sink_length: int = 0,
    bits: int = 2,
    batch_size: int = 1,
    num_hidden_layers: int = 36,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
) -> ByteReport:
    """Return KIVI active bytes with blockwise Key and rolling Value residuals."""

    _require_non_negative("seq_len", seq_len)
    _require_non_negative("sink_length", sink_length)
    _require_shape(batch_size, num_hidden_layers, num_key_value_heads, head_dim)
    for name, value in (
        ("group_size", group_size),
        ("residual_length", residual_length),
        ("bits", bits),
        ("bytes_per_meta", bytes_per_meta),
        ("bytes_per_full_precision", bytes_per_full_precision),
    ):
        _require_positive(name, value)
    if residual_length % group_size != 0:
        raise ValueError("residual_length must be a multiple of group_size")
    if head_dim % group_size != 0:
        raise ValueError("head_dim must be divisible by group_size")

    sink_tokens = min(seq_len, sink_length)
    non_sink_tokens = seq_len - sink_tokens
    key_quantized_tokens = non_sink_tokens - non_sink_tokens % residual_length
    key_residual_tokens = non_sink_tokens - key_quantized_tokens
    value_quantized_tokens = max(0, non_sink_tokens - residual_length)
    value_residual_tokens = non_sink_tokens - value_quantized_tokens
    values_per_token = batch_size * num_key_value_heads * head_dim

    key_meta_elements = (
        batch_size
        * num_key_value_heads
        * head_dim
        * _ceil_div(key_quantized_tokens, group_size)
    )
    value_meta_elements = (
        batch_size
        * num_key_value_heads
        * value_quantized_tokens
        * _ceil_div(head_dim, group_size)
    )
    per_layer = {
        "key_payload_bytes": _packed_bytes(values_per_token * key_quantized_tokens, bits),
        "value_payload_bytes": _packed_bytes(values_per_token * value_quantized_tokens, bits),
        "key_scale_bytes": key_meta_elements * bytes_per_meta,
        "key_min_or_zero_point_bytes": key_meta_elements * bytes_per_meta,
        "value_scale_bytes": value_meta_elements * bytes_per_meta,
        "value_min_or_zero_point_bytes": value_meta_elements * bytes_per_meta,
        "sink_fp16_bytes": 2
        * values_per_token
        * sink_tokens
        * bytes_per_full_precision,
        "key_residual_fp16_bytes": values_per_token
        * key_residual_tokens
        * bytes_per_full_precision,
        "value_residual_fp16_bytes": values_per_token
        * value_residual_tokens
        * bytes_per_full_precision,
    }
    method = f"kivi-g{group_size}-r{residual_length}"
    if sink_length:
        method += f"-sink{sink_length}"
    return _report(
        method=method,
        seq_len=seq_len,
        num_hidden_layers=num_hidden_layers,
        token_state={
            "sink_tokens": sink_tokens,
            "key_quantized_tokens": key_quantized_tokens,
            "key_fp16_tokens": sink_tokens + key_residual_tokens,
            "value_quantized_tokens": value_quantized_tokens,
            "value_fp16_tokens": sink_tokens + value_residual_tokens,
        },
        per_layer=per_layer,
    )


def estimate_qwen3_cage_bytes(
    *,
    seq_len: int,
    residual_length: int,
    key_bucket_sizes: Sequence[int] = (42, 43, 43),
    value_bucket_sizes: Sequence[int] = (42, 43, 43),
    key_group_sizes: Sequence[int] = (32, 64, 128),
    value_group_sizes: Sequence[int] = (32, 64, 128),
    bits: int = 2,
    batch_size: int = 1,
    num_hidden_layers: int = 36,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
    bytes_per_bucket_index: int = 8,
) -> ByteReport:
    """Return conservative CAGE bytes for the frozen separate K/V index layout."""

    _require_non_negative("seq_len", seq_len)
    _require_shape(batch_size, num_hidden_layers, num_key_value_heads, head_dim)
    for name, value in (
        ("residual_length", residual_length),
        ("bits", bits),
        ("bytes_per_meta", bytes_per_meta),
        ("bytes_per_full_precision", bytes_per_full_precision),
        ("bytes_per_bucket_index", bytes_per_bucket_index),
    ):
        _require_positive(name, value)
    key_bucket_sizes, key_group_sizes = _validate_buckets(
        "key", key_bucket_sizes, key_group_sizes, head_dim
    )
    value_bucket_sizes, value_group_sizes = _validate_buckets(
        "value", value_bucket_sizes, value_group_sizes, head_dim
    )

    key_quantized_tokens = seq_len - seq_len % residual_length
    key_residual_tokens = seq_len - key_quantized_tokens
    value_quantized_tokens = max(0, seq_len - residual_length)
    value_residual_tokens = seq_len - value_quantized_tokens
    values_per_token = batch_size * num_key_value_heads * head_dim
    key_meta_elements = sum(
        batch_size
        * num_key_value_heads
        * bucket_size
        * _ceil_div(key_quantized_tokens, group_size)
        for bucket_size, group_size in zip(key_bucket_sizes, key_group_sizes)
    )
    value_meta_elements = sum(
        batch_size
        * num_key_value_heads
        * value_quantized_tokens
        * _ceil_div(bucket_size, group_size)
        for bucket_size, group_size in zip(value_bucket_sizes, value_group_sizes)
    )
    per_layer = {
        "key_payload_bytes": _packed_bytes(values_per_token * key_quantized_tokens, bits),
        "value_payload_bytes": _packed_bytes(values_per_token * value_quantized_tokens, bits),
        "key_scale_bytes": key_meta_elements * bytes_per_meta,
        "key_min_or_zero_point_bytes": key_meta_elements * bytes_per_meta,
        "value_scale_bytes": value_meta_elements * bytes_per_meta,
        "value_min_or_zero_point_bytes": value_meta_elements * bytes_per_meta,
        "bucket_index_bytes": 2
        * num_key_value_heads
        * head_dim
        * bytes_per_bucket_index,
        "key_residual_fp16_bytes": values_per_token
        * key_residual_tokens
        * bytes_per_full_precision,
        "value_residual_fp16_bytes": values_per_token
        * value_residual_tokens
        * bytes_per_full_precision,
    }
    return _report(
        method=f"cage-r{residual_length}",
        seq_len=seq_len,
        num_hidden_layers=num_hidden_layers,
        token_state={
            "key_quantized_tokens": key_quantized_tokens,
            "key_fp16_tokens": key_residual_tokens,
            "value_quantized_tokens": value_quantized_tokens,
            "value_fp16_tokens": value_residual_tokens,
        },
        per_layer=per_layer,
    )


def estimate_qwen3_kitty_bytes(
    *,
    seq_len: int,
    boosted_channels: int,
    page_size: int = 128,
    sink_length: int = 32,
    low_bits: int = 2,
    batch_size: int = 1,
    num_hidden_layers: int = 36,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
    bytes_per_channel_index: int = 1,
    bytes_per_page_table_entry: int = 8,
) -> ByteReport:
    """Return active Kitty bytes from the frozen official packed page layout."""

    _require_non_negative("seq_len", seq_len)
    _require_non_negative("boosted_channels", boosted_channels)
    _require_non_negative("sink_length", sink_length)
    _require_shape(batch_size, num_hidden_layers, num_key_value_heads, head_dim)
    for name, value in (
        ("page_size", page_size),
        ("low_bits", low_bits),
        ("bytes_per_meta", bytes_per_meta),
        ("bytes_per_full_precision", bytes_per_full_precision),
        ("bytes_per_channel_index", bytes_per_channel_index),
        ("bytes_per_page_table_entry", bytes_per_page_table_entry),
    ):
        _require_positive(name, value)
    if boosted_channels > head_dim:
        raise ValueError("boosted_channels must not exceed head_dim")
    if page_size % 128 != 0:
        raise ValueError("the frozen Kitty runtime requires page_size to be a multiple of 128")

    sink_tokens = min(seq_len, sink_length)
    non_sink_tokens = seq_len - sink_tokens
    key_buffer_tokens = non_sink_tokens % page_size
    key_page_count = (non_sink_tokens - key_buffer_tokens) // page_size
    value_local_tokens = min(page_size, non_sink_tokens)
    value_nonlocal_tokens = non_sink_tokens - value_local_tokens
    value_buffer_tokens = value_nonlocal_tokens % page_size
    value_page_count = (value_nonlocal_tokens - value_buffer_tokens) // page_size

    key_low_payload_per_page = _packed_bytes(
        num_key_value_heads * head_dim * page_size, low_bits
    )
    key_promoted_high_payload_per_page = _packed_bytes(
        num_key_value_heads * boosted_channels * page_size, low_bits
    )
    key_channel_index_per_page = (
        num_key_value_heads * head_dim * bytes_per_channel_index
    )
    key_scale_per_page = num_key_value_heads * head_dim * bytes_per_meta
    key_zero_point_per_page = key_scale_per_page
    value_payload_per_page = _packed_bytes(
        num_key_value_heads * page_size * head_dim, low_bits
    )
    value_scale_per_page = num_key_value_heads * page_size * bytes_per_meta
    value_zero_point_per_page = value_scale_per_page
    key_page_bytes = (
        key_low_payload_per_page
        + key_promoted_high_payload_per_page
        + key_channel_index_per_page
        + key_scale_per_page
        + key_zero_point_per_page
    )
    value_page_bytes = (
        value_payload_per_page + value_scale_per_page + value_zero_point_per_page
    )
    values_per_token = batch_size * num_key_value_heads * head_dim
    per_layer = {
        "key_low_payload_bytes": batch_size
        * key_page_count
        * key_low_payload_per_page,
        "key_promoted_high_payload_bytes": batch_size
        * key_page_count
        * key_promoted_high_payload_per_page,
        "key_channel_index_bytes": batch_size
        * key_page_count
        * key_channel_index_per_page,
        "key_scale_bytes": batch_size * key_page_count * key_scale_per_page,
        "key_zero_point_bytes": batch_size * key_page_count * key_zero_point_per_page,
        "value_payload_bytes": batch_size * value_page_count * value_payload_per_page,
        "value_scale_bytes": batch_size * value_page_count * value_scale_per_page,
        "value_zero_point_bytes": batch_size
        * value_page_count
        * value_zero_point_per_page,
        "page_table_bytes": batch_size
        * (key_page_count + value_page_count)
        * bytes_per_page_table_entry,
        "sink_fp16_bytes": 2
        * values_per_token
        * sink_tokens
        * bytes_per_full_precision,
        "key_buffer_fp16_bytes": values_per_token
        * key_buffer_tokens
        * bytes_per_full_precision,
        "value_buffer_fp16_bytes": values_per_token
        * value_buffer_tokens
        * bytes_per_full_precision,
        "value_local_fp16_bytes": values_per_token
        * value_local_tokens
        * bytes_per_full_precision,
    }
    if boosted_channels * 8 == head_dim:
        method = "kitty-12.5pct"
    elif boosted_channels * 4 == head_dim:
        method = "kitty-pro-25pct"
    else:
        method = f"kitty-boost{boosted_channels}"
    report = _report(
        method=method,
        seq_len=seq_len,
        num_hidden_layers=num_hidden_layers,
        token_state={
            "sink_tokens": sink_tokens,
            "key_page_count": key_page_count,
            "key_buffer_tokens": key_buffer_tokens,
            "value_page_count": value_page_count,
            "value_buffer_tokens": value_buffer_tokens,
            "value_local_tokens": value_local_tokens,
        },
        per_layer=per_layer,
    )
    report["page_layout"] = {
        "key_page_bytes": key_page_bytes,
        "value_page_bytes": value_page_bytes,
        "combined_page_bytes": key_page_bytes + value_page_bytes,
        "steady_bytes_per_token_model_wide": (key_page_bytes + value_page_bytes)
        * num_hidden_layers
        / page_size,
    }
    return report


def closest_memory_candidate(
    candidates: Iterable[ByteReport], target_bytes: int
) -> tuple[ByteReport, float]:
    """Select by the frozen byte-only rule, without consuming quality results."""

    _require_positive("target_bytes", target_bytes)
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")

    def tie_key(report: ByteReport) -> tuple[float, int, int, int]:
        total = int(report["model_total_bytes"])
        residual = int(report.get("residual_length", 0))
        group = int(report.get("group_size", 0))
        return (abs(total - target_bytes) / target_bytes, total, residual, group)

    selected = min(candidates, key=tie_key)
    relative_delta = (int(selected["model_total_bytes"]) - target_bytes) / target_bytes
    return selected, relative_delta


def _report(
    *,
    method: str,
    seq_len: int,
    num_hidden_layers: int,
    token_state: Mapping[str, int],
    per_layer: Mapping[str, int],
) -> ByteReport:
    normalized = {name: int(value) for name, value in per_layer.items()}
    normalized["total_bytes"] = sum(normalized.values())
    report: ByteReport = {
        "method": method,
        "seq_len": seq_len,
        "token_state": dict(token_state),
        "per_layer": normalized,
        "model_total_bytes": normalized["total_bytes"] * num_hidden_layers,
    }
    if method.startswith("cage-r"):
        report["residual_length"] = int(method.split("r", 1)[1])
    elif method.startswith("kivi-g"):
        core = method.split("-sink", 1)[0]
        group_text, residual_text = core.removeprefix("kivi-g").split("-r")
        report["group_size"] = int(group_text)
        report["residual_length"] = int(residual_text)
    return report


def _validate_buckets(
    name: str,
    bucket_sizes: Sequence[int],
    group_sizes: Sequence[int],
    head_dim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    bucket_sizes = tuple(int(value) for value in bucket_sizes)
    group_sizes = tuple(int(value) for value in group_sizes)
    if not bucket_sizes or len(bucket_sizes) != len(group_sizes):
        raise ValueError(f"{name} bucket and group sizes must be nonempty and aligned")
    if sum(bucket_sizes) != head_dim:
        raise ValueError(f"{name} bucket sizes must sum to head_dim={head_dim}")
    for index, value in enumerate(bucket_sizes):
        _require_positive(f"{name}_bucket_sizes[{index}]", value)
    for index, value in enumerate(group_sizes):
        _require_positive(f"{name}_group_sizes[{index}]", value)
    return bucket_sizes, group_sizes


def _packed_bytes(num_values: int, bits: int) -> int:
    return _ceil_div(num_values * bits, 8)


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator) if numerator else 0


def _require_shape(
    batch_size: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("num_hidden_layers", num_hidden_layers),
        ("num_key_value_heads", num_key_value_heads),
        ("head_dim", head_dim),
    ):
        _require_positive(name, value)


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


__all__ = [
    "closest_memory_candidate",
    "estimate_qwen3_cage_bytes",
    "estimate_qwen3_fp16_bytes",
    "estimate_qwen3_kivi_bytes",
    "estimate_qwen3_kitty_bytes",
]
