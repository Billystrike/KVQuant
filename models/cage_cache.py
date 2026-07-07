from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CAGE_PAST_KEY_VALUE_TAG = "CAGE_KV_CACHE_V1"


@dataclass(frozen=True)
class CageKeyCache:
    key_quant_buckets: tuple[Any, ...] = ()
    key_full: Any | None = None
    key_bucket_indices: tuple[Any, ...] = ()
    key_group_sizes: tuple[int, ...] = ()
    key_clip_percentiles: tuple[float, ...] = ()
    key_scales: tuple[Any, ...] | None = None
    key_mins: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_quant_buckets", _as_tuple(self.key_quant_buckets))
        object.__setattr__(self, "key_bucket_indices", _as_tuple(self.key_bucket_indices))
        object.__setattr__(self, "key_group_sizes", tuple(self.key_group_sizes))
        object.__setattr__(self, "key_clip_percentiles", tuple(self.key_clip_percentiles))
        object.__setattr__(self, "key_scales", _as_optional_tuple(self.key_scales))
        object.__setattr__(self, "key_mins", _as_optional_tuple(self.key_mins))

    def bucket_indices_like(self, tensor: Any) -> tuple[Any, ...]:
        return move_bucket_indices_like(self.key_bucket_indices, tensor)


@dataclass(frozen=True)
class CageValueCache:
    value_quant_buckets: tuple[Any, ...] = ()
    value_full: Any | None = None
    value_bucket_indices: tuple[Any, ...] = ()
    value_group_sizes: tuple[int, ...] = ()
    value_clip_percentiles: tuple[float, ...] = ()
    value_scales: tuple[Any, ...] | None = None
    value_mins: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_quant_buckets", _as_tuple(self.value_quant_buckets))
        object.__setattr__(self, "value_bucket_indices", _as_tuple(self.value_bucket_indices))
        object.__setattr__(self, "value_group_sizes", tuple(self.value_group_sizes))
        object.__setattr__(self, "value_clip_percentiles", tuple(self.value_clip_percentiles))
        object.__setattr__(self, "value_scales", _as_optional_tuple(self.value_scales))
        object.__setattr__(self, "value_mins", _as_optional_tuple(self.value_mins))

    def bucket_indices_like(self, tensor: Any) -> tuple[Any, ...]:
        return move_bucket_indices_like(self.value_bucket_indices, tensor)


@dataclass(frozen=True)
class CagePastKeyValue:
    key_cache: CageKeyCache
    value_cache: CageValueCache
    kv_seq_len: int

    def __post_init__(self) -> None:
        if self.kv_seq_len < 0:
            raise ValueError(f"kv_seq_len must be non-negative, got {self.kv_seq_len}")

    def to_legacy_tuple(self) -> tuple[str, CageKeyCache, CageValueCache, int]:
        return (CAGE_PAST_KEY_VALUE_TAG, self.key_cache, self.value_cache, self.kv_seq_len)


def is_cage_past_key_value(past_key_value: Any) -> bool:
    if isinstance(past_key_value, CagePastKeyValue):
        return True
    return (
        isinstance(past_key_value, tuple)
        and len(past_key_value) == 4
        and past_key_value[0] == CAGE_PAST_KEY_VALUE_TAG
    )


def pack_cage_past_key_value(
    key_cache: CageKeyCache,
    value_cache: CageValueCache,
    kv_seq_len: int | None = None,
) -> tuple[str, CageKeyCache, CageValueCache, int]:
    inferred_kv_seq_len = infer_cage_kv_seq_len(key_cache, value_cache)
    if kv_seq_len is None:
        kv_seq_len = inferred_kv_seq_len
    elif inferred_kv_seq_len != 0 and kv_seq_len != inferred_kv_seq_len:
        raise ValueError(
            f"kv_seq_len={kv_seq_len} does not match inferred cache length {inferred_kv_seq_len}"
        )

    return CagePastKeyValue(
        key_cache=key_cache,
        value_cache=value_cache,
        kv_seq_len=kv_seq_len,
    ).to_legacy_tuple()


def unpack_cage_past_key_value(past_key_value: Any) -> CagePastKeyValue:
    if isinstance(past_key_value, CagePastKeyValue):
        return past_key_value

    if not is_cage_past_key_value(past_key_value):
        raise ValueError("past_key_value is not a CAGE past_key_value")

    _, key_cache, value_cache, kv_seq_len = past_key_value
    if not isinstance(key_cache, CageKeyCache) or not isinstance(value_cache, CageValueCache):
        raise ValueError("malformed CAGE past_key_value cache records")

    return CagePastKeyValue(
        key_cache=key_cache,
        value_cache=value_cache,
        kv_seq_len=kv_seq_len,
    )


def infer_cage_kv_seq_len(key_cache: CageKeyCache, value_cache: CageValueCache) -> int:
    key_seq_len = _infer_side_seq_len(key_cache.key_quant_buckets, key_cache.key_full)
    value_seq_len = _infer_side_seq_len(value_cache.value_quant_buckets, value_cache.value_full)
    known_lengths = [length for length in (key_seq_len, value_seq_len) if length is not None]

    if not known_lengths:
        return 0
    if len(set(known_lengths)) != 1:
        raise ValueError(
            f"Key and Value cache lengths must match, got {key_seq_len} and {value_seq_len}"
        )
    return known_lengths[0]


def move_bucket_indices_like(bucket_indices: Any, tensor: Any) -> tuple[Any, ...]:
    device = getattr(tensor, "device", tensor)
    return tuple(_move_to_device(index, device) for index in _as_tuple(bucket_indices))


def _infer_side_seq_len(quant_buckets: tuple[Any, ...], full_tensor: Any | None) -> int | None:
    quant_lengths = {
        length
        for length in (_sequence_length(bucket) for bucket in quant_buckets)
        if length is not None
    }
    if len(quant_lengths) > 1:
        raise ValueError(f"Bucket payload sequence lengths must match, got {sorted(quant_lengths)}")

    full_length = _sequence_length(full_tensor)
    if not quant_lengths and full_length is None:
        return None

    quant_length = next(iter(quant_lengths), 0)
    return quant_length + (full_length or 0)


def _sequence_length(tensor: Any | None) -> int | None:
    if tensor is None:
        return None

    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[-2])


def _move_to_device(value: Any, device: Any) -> Any:
    to = getattr(value, "to", None)
    if to is None:
        return value
    return to(device=device)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _as_optional_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return _as_tuple(value)
