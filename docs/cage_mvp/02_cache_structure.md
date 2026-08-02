# Task 02: CAGE Cache Structure

## Objective

Define lightweight cache helpers for CAGE-KV without immediately rewriting all existing KIVI tuple logic.

## Scope

Add a CAGE-only cache representation that can coexist with the original `past_key_value` tuple used by KIVI.

## Required module

Create `models/cage_cache.py`.

## Key cache fields

A CAGE Key cache record should store:

- `key_quant_buckets`: quantized or fake-quantized bucket payloads.
- `key_full`: recent full-precision residual Key states.
- `key_bucket_indices`: channel indices for each bucket.
- `key_group_sizes`: group size per bucket.
- `key_clip_percentiles`: clipping percentile per bucket.
- `key_scales`: optional scale tensors for reporting/debugging.
- `key_mins`: optional min or zero-point tensors for reporting/debugging.

## Value cache fields

A CAGE Value cache record should store:

- `value_quant_buckets`: quantized or fake-quantized bucket payloads.
- `value_full`: recent full-precision residual Value states.
- `value_bucket_indices`: channel indices for each bucket.
- `value_group_sizes`: group size per bucket.
- `value_clip_percentiles`: clipping percentile per bucket.
- `value_scales`: optional scale tensors for reporting/debugging.
- `value_mins`: optional min or zero-point tensors for reporting/debugging.

## Compatibility helpers

Expose:

```python
is_cage_past_key_value(past_key_value) -> bool
pack_cage_past_key_value(...) -> tuple
unpack_cage_past_key_value(past_key_value) -> CagePastKeyValue
```

The packed format may remain a tuple for compatibility with generation internals, but it must be self-identifying so it is not confused with the original KIVI 9-item tuple.

## Design requirements

1. `kv_seq_len` must be preserved and easy to read.
2. Original KIVI past-key-value tuples must remain valid.
3. The MVP fake path may store dequantized fake-quant tensors as bucket payloads, but the structure should be extensible to packed INT2 records later.
4. Bucket indices must remain on the same device as the associated tensors or be moved safely before index operations.

## Acceptance criteria

- A CAGE cache can round-trip through pack/unpack helpers.
- `is_cage_past_key_value` returns `False` for original KIVI tuples.
- `kv_seq_len` equals quantized history length plus residual length.
- No model forward path is changed in this task beyond imports if needed.
