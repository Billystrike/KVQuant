from __future__ import annotations

import math
from typing import Any, Sequence, TypedDict

from models.cage_cache import is_cage_past_key_value, unpack_cage_past_key_value


_REQUIRED_BYTE_FIELDS = (
    "key_payload_bytes",
    "value_payload_bytes",
    "key_scale_bytes",
    "value_scale_bytes",
    "key_min_or_zp_bytes",
    "value_min_or_zp_bytes",
    "bucket_index_bytes",
    "residual_full_precision_bytes",
)


CacheSummary = dict[str, int | str]


class CudaPeakDiagnostic(TypedDict):
    max_allocated_bytes: int
    max_reserved_bytes: int


class MemoryNamespace(TypedDict):
    paper_estimate: CacheSummary
    runtime_tensors: CacheSummary
    cuda_peak_diagnostic: CudaPeakDiagnostic


def build_memory_namespace(
    paper_summary: CacheSummary,
    runtime_summary: CacheSummary,
    *,
    max_allocated_bytes: int,
    max_reserved_bytes: int,
) -> MemoryNamespace:
    """Build the exact worker-facing cache memory namespace."""

    return {
        "paper_estimate": paper_summary,
        "runtime_tensors": runtime_summary,
        "cuda_peak_diagnostic": {
            "max_allocated_bytes": int(max_allocated_bytes),
            "max_reserved_bytes": int(max_reserved_bytes),
        },
    }


def estimate_kivi_cache_bytes(
    *,
    batch_size: int,
    num_key_value_heads: int,
    seq_len: int,
    head_dim: int,
    group_size: int,
    residual_length: int = 0,
    bits: int = 2,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
) -> dict[str, int | str]:
    """Estimate paper-facing KIVI cache bytes for one layer."""

    _require_non_negative("seq_len", seq_len)
    _require_non_negative("residual_length", residual_length)
    _require_positive("batch_size", batch_size)
    _require_positive("num_key_value_heads", num_key_value_heads)
    _require_positive("head_dim", head_dim)
    _require_positive("group_size", group_size)
    _require_positive("bits", bits)
    _require_positive("bytes_per_meta", bytes_per_meta)
    _require_positive("bytes_per_full_precision", bytes_per_full_precision)

    residual_tokens = min(seq_len, residual_length)
    quant_tokens = max(seq_len - residual_tokens, 0)
    payload_shape_values = batch_size * num_key_value_heads * quant_tokens * head_dim

    key_payload_bytes = _packed_int_payload_bytes(payload_shape_values, bits)
    value_payload_bytes = _packed_int_payload_bytes(payload_shape_values, bits)
    key_meta_elements = batch_size * num_key_value_heads * head_dim * _ceil_div(quant_tokens, group_size)
    value_meta_elements = batch_size * num_key_value_heads * quant_tokens * _ceil_div(head_dim, group_size)
    residual_full_precision_bytes = (
        2
        * batch_size
        * num_key_value_heads
        * residual_tokens
        * head_dim
        * bytes_per_full_precision
    )

    return _finalize_summary(
        cache_type="kivi_estimate",
        key_payload_bytes=key_payload_bytes,
        value_payload_bytes=value_payload_bytes,
        key_scale_bytes=key_meta_elements * bytes_per_meta,
        value_scale_bytes=value_meta_elements * bytes_per_meta,
        key_min_or_zp_bytes=key_meta_elements * bytes_per_meta,
        value_min_or_zp_bytes=value_meta_elements * bytes_per_meta,
        bucket_index_bytes=0,
        residual_full_precision_bytes=residual_full_precision_bytes,
    )


def estimate_cage_cache_bytes(
    *,
    batch_size: int,
    num_key_value_heads: int,
    seq_len: int,
    head_dim: int,
    key_bucket_sizes: Sequence[int] | None = None,
    value_bucket_sizes: Sequence[int] | None = None,
    key_group_sizes: Sequence[int] = (32, 64, 128),
    value_group_sizes: Sequence[int] = (32, 64, 128),
    residual_length: int = 0,
    bits: int = 2,
    bytes_per_meta: int = 2,
    bytes_per_full_precision: int = 2,
    bucket_index_bytes: int = 0,
) -> dict[str, int | str]:
    """Estimate paper-facing CAGE-KV cache bytes for one layer."""

    _require_non_negative("seq_len", seq_len)
    _require_non_negative("residual_length", residual_length)
    _require_non_negative("bucket_index_bytes", bucket_index_bytes)
    _require_positive("batch_size", batch_size)
    _require_positive("num_key_value_heads", num_key_value_heads)
    _require_positive("head_dim", head_dim)
    _require_positive("bits", bits)
    _require_positive("bytes_per_meta", bytes_per_meta)
    _require_positive("bytes_per_full_precision", bytes_per_full_precision)

    key_bucket_sizes = _normalize_bucket_sizes("key_bucket_sizes", key_bucket_sizes, head_dim)
    value_bucket_sizes = _normalize_bucket_sizes("value_bucket_sizes", value_bucket_sizes, head_dim)
    _require_group_sizes("key_group_sizes", key_group_sizes, len(key_bucket_sizes))
    _require_group_sizes("value_group_sizes", value_group_sizes, len(value_bucket_sizes))

    residual_tokens = min(seq_len, residual_length)
    quant_tokens = max(seq_len - residual_tokens, 0)

    key_payload_bytes = sum(
        _packed_int_payload_bytes(batch_size * num_key_value_heads * quant_tokens * bucket_size, bits)
        for bucket_size in key_bucket_sizes
    )
    value_payload_bytes = sum(
        _packed_int_payload_bytes(batch_size * num_key_value_heads * quant_tokens * bucket_size, bits)
        for bucket_size in value_bucket_sizes
    )

    key_scale_bytes = _estimate_cage_key_meta_bytes(
        batch_size=batch_size,
        num_key_value_heads=num_key_value_heads,
        quant_tokens=quant_tokens,
        bucket_sizes=key_bucket_sizes,
        group_sizes=key_group_sizes,
        bytes_per_meta=bytes_per_meta,
    )
    value_scale_bytes = _estimate_cage_value_meta_bytes(
        batch_size=batch_size,
        num_key_value_heads=num_key_value_heads,
        quant_tokens=quant_tokens,
        bucket_sizes=value_bucket_sizes,
        group_sizes=value_group_sizes,
        bytes_per_meta=bytes_per_meta,
    )
    residual_full_precision_bytes = (
        2
        * batch_size
        * num_key_value_heads
        * residual_tokens
        * head_dim
        * bytes_per_full_precision
    )

    return _finalize_summary(
        cache_type="cage_estimate",
        key_payload_bytes=key_payload_bytes,
        value_payload_bytes=value_payload_bytes,
        key_scale_bytes=key_scale_bytes,
        value_scale_bytes=value_scale_bytes,
        key_min_or_zp_bytes=key_scale_bytes,
        value_min_or_zp_bytes=value_scale_bytes,
        bucket_index_bytes=bucket_index_bytes,
        residual_full_precision_bytes=residual_full_precision_bytes,
    )


def summarize_cache_bytes(past_key_value: Any) -> dict[str, int | str]:
    """Summarize one layer of KIVI or CAGE cache bytes in a JSON-safe dict."""

    if past_key_value is None:
        return _finalize_summary(cache_type="empty")
    if is_cage_past_key_value(past_key_value):
        return _summarize_cage_cache_bytes(past_key_value)
    if isinstance(past_key_value, tuple) and len(past_key_value) >= 9:
        return _summarize_kivi_tuple_bytes(past_key_value)
    raise ValueError("past_key_value must be a CAGE cache or original KIVI cache tuple")


def summarize_runtime_cache_bytes(past_key_value: Any) -> dict[str, int | str]:
    """Summarize tensor bytes actually retained by one runtime cache layer."""

    if past_key_value is None:
        return _finalize_summary(cache_type="empty_runtime")
    if is_cage_past_key_value(past_key_value):
        unpacked = unpack_cage_past_key_value(past_key_value)
        key_cache = unpacked.key_cache
        value_cache = unpacked.value_cache
        return _finalize_summary(
            cache_type="cage_fake_runtime",
            key_payload_bytes=_tensor_sequence_nbytes(key_cache.key_quant_buckets),
            value_payload_bytes=_tensor_sequence_nbytes(value_cache.value_quant_buckets),
            bucket_index_bytes=_tensor_sequence_nbytes(key_cache.key_bucket_indices)
            + _tensor_sequence_nbytes(value_cache.value_bucket_indices),
            residual_full_precision_bytes=_tensor_nbytes(key_cache.key_full)
            + _tensor_nbytes(value_cache.value_full),
        )
    if isinstance(past_key_value, tuple) and len(past_key_value) >= 9:
        summary = _summarize_kivi_tuple_bytes(past_key_value)
        summary["cache_type"] = "kivi_runtime"
        return summary
    raise ValueError("past_key_value must be a CAGE cache or original KIVI cache tuple")


def summarize_fp16_cache_bytes(past_key_value: tuple[Any, Any]) -> dict[str, int | str]:
    """Summarize a conventional FP16 key/value cache layer."""

    if not isinstance(past_key_value, tuple) or len(past_key_value) != 2:
        raise ValueError("FP16 past_key_value must be a (key, value) tuple")
    return _finalize_summary(
        cache_type="fp16",
        residual_full_precision_bytes=_tensor_nbytes(past_key_value[0])
        + _tensor_nbytes(past_key_value[1]),
    )


def summarize_cache_structure(
    method: str,
    past_key_value: Any,
    *,
    prompt_length: int,
    residual_length: int | None = None,
) -> dict[str, dict[str, int]]:
    """Audit Key/Value token partitions for one prefill cache layer."""

    _require_positive("prompt_length", prompt_length)
    if method == "fp16":
        if not isinstance(past_key_value, tuple) or len(past_key_value) != 2:
            raise ValueError("FP16 cache must be a (key, value) tuple")
        key_total = _sequence_length(past_key_value[0])
        value_total = _sequence_length(past_key_value[1])
        if key_total != prompt_length or value_total != prompt_length:
            raise ValueError("FP16 Key and Value total lengths must equal prompt_length")
        key_residual = value_residual = prompt_length
    elif method == "kivi":
        if not isinstance(past_key_value, tuple) or len(past_key_value) < 9:
            raise ValueError("KIVI cache must use the original packed tuple")
        _require_positive("residual_length", residual_length)
        if past_key_value[8] != prompt_length:
            raise ValueError("KIVI Key and Value total lengths must equal prompt_length")
        key_residual = _sequence_length(past_key_value[1])
        value_residual = _sequence_length(past_key_value[5])
    elif method == "cage":
        _require_positive("residual_length", residual_length)
        unpacked = unpack_cage_past_key_value(past_key_value)
        if unpacked.kv_seq_len != prompt_length:
            raise ValueError("CAGE Key and Value total lengths must equal prompt_length")
        key_residual = _sequence_length(unpacked.key_cache.key_full)
        value_residual = _sequence_length(unpacked.value_cache.value_full)
        _require_bucket_sequence_length(
            "CAGE Key", unpacked.key_cache.key_quant_buckets,
            prompt_length - key_residual,
        )
        _require_bucket_sequence_length(
            "CAGE Value", unpacked.value_cache.value_quant_buckets,
            prompt_length - value_residual,
        )
    else:
        raise ValueError(f"unsupported cache method {method!r}")

    if method in {"kivi", "cage"}:
        expected_key_residual = prompt_length % int(residual_length)
        expected_value_residual = min(prompt_length, int(residual_length))
        if key_residual != expected_key_residual:
            raise ValueError(
                f"Key residual tokens must equal T % residual_length = "
                f"{expected_key_residual}, got {key_residual}"
            )
        if value_residual != expected_value_residual:
            raise ValueError(
                f"Value residual tokens must equal min(T, residual_length) = "
                f"{expected_value_residual}, got {value_residual}"
            )

    return {
        "key": {
            "total_tokens": prompt_length,
            "quantized_history_tokens": prompt_length - key_residual,
            "fp16_residual_tokens": key_residual,
        },
        "value": {
            "total_tokens": prompt_length,
            "quantized_history_tokens": prompt_length - value_residual,
            "fp16_residual_tokens": value_residual,
        },
    }


def sum_cache_summaries(layer_summaries: Sequence[dict[str, int | str]]) -> dict[str, int | str]:
    """Add every numeric byte field present across layer summaries."""

    total: dict[str, int | str] = {"cache_type": "model_total"}
    for summary in layer_summaries:
        for field, value in summary.items():
            if field != "cache_type" and isinstance(value, int):
                total[field] = int(total.get(field, 0)) + value
    return total


def _summarize_cage_cache_bytes(past_key_value: Any) -> dict[str, int | str]:
    unpacked = unpack_cage_past_key_value(past_key_value)
    key_cache = unpacked.key_cache
    value_cache = unpacked.value_cache

    key_payload_bytes = sum(
        _packed_int_payload_bytes(_numel(bucket), bits=2)
        for bucket in key_cache.key_quant_buckets
        if bucket is not None
    )
    value_payload_bytes = sum(
        _packed_int_payload_bytes(_numel(bucket), bits=2)
        for bucket in value_cache.value_quant_buckets
        if bucket is not None
    )
    key_scale_bytes = _summarize_cage_key_meta_bytes(key_cache.key_quant_buckets, key_cache.key_group_sizes)
    value_scale_bytes = _summarize_cage_value_meta_bytes(value_cache.value_quant_buckets, value_cache.value_group_sizes)

    return _finalize_summary(
        cache_type="cage",
        key_payload_bytes=key_payload_bytes,
        value_payload_bytes=value_payload_bytes,
        key_scale_bytes=key_scale_bytes,
        value_scale_bytes=value_scale_bytes,
        key_min_or_zp_bytes=key_scale_bytes,
        value_min_or_zp_bytes=value_scale_bytes,
        bucket_index_bytes=_tensor_sequence_nbytes(key_cache.key_bucket_indices)
        + _tensor_sequence_nbytes(value_cache.value_bucket_indices),
        residual_full_precision_bytes=_tensor_nbytes(key_cache.key_full)
        + _tensor_nbytes(value_cache.value_full),
    )


def _summarize_kivi_tuple_bytes(past_key_value: tuple[Any, ...]) -> dict[str, int | str]:
    return _finalize_summary(
        cache_type="kivi",
        key_payload_bytes=_tensor_nbytes(past_key_value[0]),
        value_payload_bytes=_tensor_nbytes(past_key_value[4]),
        key_scale_bytes=_tensor_nbytes(past_key_value[2]),
        value_scale_bytes=_tensor_nbytes(past_key_value[6]),
        key_min_or_zp_bytes=_tensor_nbytes(past_key_value[3]),
        value_min_or_zp_bytes=_tensor_nbytes(past_key_value[7]),
        bucket_index_bytes=0,
        residual_full_precision_bytes=_tensor_nbytes(past_key_value[1])
        + _tensor_nbytes(past_key_value[5]),
    )


def _summarize_cage_key_meta_bytes(quant_buckets: Sequence[Any], group_sizes: Sequence[int]) -> int:
    _require_group_sizes("key_group_sizes", group_sizes, len(quant_buckets))
    total = 0
    for bucket, group_size in zip(quant_buckets, group_sizes):
        if bucket is None:
            continue
        batch_size, heads, quant_tokens, bucket_size = _bucket_shape(bucket)
        total += batch_size * heads * bucket_size * _ceil_div(quant_tokens, group_size) * 2
    return total


def _summarize_cage_value_meta_bytes(quant_buckets: Sequence[Any], group_sizes: Sequence[int]) -> int:
    _require_group_sizes("value_group_sizes", group_sizes, len(quant_buckets))
    total = 0
    for bucket, group_size in zip(quant_buckets, group_sizes):
        if bucket is None:
            continue
        batch_size, heads, quant_tokens, bucket_size = _bucket_shape(bucket)
        total += batch_size * heads * quant_tokens * _ceil_div(bucket_size, group_size) * 2
    return total


def _estimate_cage_key_meta_bytes(
    *,
    batch_size: int,
    num_key_value_heads: int,
    quant_tokens: int,
    bucket_sizes: Sequence[int],
    group_sizes: Sequence[int],
    bytes_per_meta: int,
) -> int:
    return sum(
        batch_size
        * num_key_value_heads
        * bucket_size
        * _ceil_div(quant_tokens, group_size)
        * bytes_per_meta
        for bucket_size, group_size in zip(bucket_sizes, group_sizes)
    )


def _estimate_cage_value_meta_bytes(
    *,
    batch_size: int,
    num_key_value_heads: int,
    quant_tokens: int,
    bucket_sizes: Sequence[int],
    group_sizes: Sequence[int],
    bytes_per_meta: int,
) -> int:
    return sum(
        batch_size
        * num_key_value_heads
        * quant_tokens
        * _ceil_div(bucket_size, group_size)
        * bytes_per_meta
        for bucket_size, group_size in zip(bucket_sizes, group_sizes)
    )


def _finalize_summary(cache_type: str, **fields: int) -> dict[str, int | str]:
    summary = {field: int(fields.get(field, 0)) for field in _REQUIRED_BYTE_FIELDS}
    summary["payload_only_bytes"] = int(summary["key_payload_bytes"] + summary["value_payload_bytes"])
    summary["metadata_bytes"] = int(
        summary["key_scale_bytes"]
        + summary["value_scale_bytes"]
        + summary["key_min_or_zp_bytes"]
        + summary["value_min_or_zp_bytes"]
    )
    summary["total_bytes"] = int(sum(summary[field] for field in _REQUIRED_BYTE_FIELDS))
    summary["cache_type"] = cache_type
    return summary


def _normalize_bucket_sizes(name: str, bucket_sizes: Sequence[int] | None, head_dim: int) -> tuple[int, ...]:
    if bucket_sizes is None:
        return (head_dim,)
    bucket_sizes = tuple(int(size) for size in bucket_sizes)
    if not bucket_sizes:
        raise ValueError(f"{name} must not be empty")
    for index, size in enumerate(bucket_sizes):
        _require_positive(f"{name}[{index}]", size)
    if sum(bucket_sizes) != head_dim:
        raise ValueError(f"{name} must sum to head_dim={head_dim}, got {sum(bucket_sizes)}")
    return bucket_sizes


def _require_group_sizes(name: str, group_sizes: Sequence[int], bucket_count: int) -> None:
    if len(group_sizes) < bucket_count:
        raise ValueError(f"{name} must have at least {bucket_count} entries, got {len(group_sizes)}")
    for index, group_size in enumerate(group_sizes[:bucket_count]):
        _require_positive(f"{name}[{index}]", int(group_size))


def _bucket_shape(bucket: Any) -> tuple[int, int, int, int]:
    shape = getattr(bucket, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError("CAGE bucket payloads must have shape [B, H_kv, T, D_bucket]")
    return tuple(int(dim) for dim in shape)


def _packed_int_payload_bytes(num_values: int, bits: int) -> int:
    _require_non_negative("num_values", num_values)
    _require_positive("bits", bits)
    return _ceil_div(num_values * bits, 8)


def _numel(tensor: Any | None) -> int:
    if tensor is None:
        return 0
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        return int(numel())
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return 0
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _tensor_sequence_nbytes(tensors: Sequence[Any] | None) -> int:
    if tensors is None:
        return 0
    return sum(_tensor_nbytes(tensor) for tensor in tensors)


def _sequence_length(tensor: Any | None) -> int:
    if tensor is None:
        return 0
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("cache tensor must expose a sequence dimension")
    return int(shape[-2])


def _require_bucket_sequence_length(
    name: str, buckets: Sequence[Any], expected: int,
) -> None:
    lengths = {_sequence_length(bucket) for bucket in buckets if bucket is not None}
    if expected == 0 and not lengths:
        return
    if lengths != {expected}:
        raise ValueError(f"{name} quantized-history lengths must equal {expected}, got {lengths}")


def _tensor_nbytes(tensor: Any | None) -> int:
    if tensor is None:
        return 0
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(tensor, "element_size", None)
    if callable(element_size):
        return int(_numel(tensor) * element_size())
    return 0


def _ceil_div(numerator: int, denominator: int) -> int:
    _require_positive("denominator", denominator)
    return math.ceil(numerator / denominator) if numerator > 0 else 0


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
