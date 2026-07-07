# Task 07: Memory Summary

## Objective

Report CAGE-KV cache memory so experiments can distinguish INT2 payload savings from extra metadata cost.

## Required module

Create `utils/cage_memory.py`.

## Required interfaces

```python
estimate_kivi_cache_bytes(...)
estimate_cage_cache_bytes(...)
summarize_cache_bytes(past_key_value) -> dict
```

Exact arguments can evolve, but the returned dictionary must be stable enough for logging.

## Required accounting fields

- `key_payload_bytes`
- `value_payload_bytes`
- `key_scale_bytes`
- `value_scale_bytes`
- `key_min_or_zp_bytes`
- `value_min_or_zp_bytes`
- `bucket_index_bytes`
- `residual_full_precision_bytes`
- `total_bytes`

## Estimation formulas

### Key metadata

For each bucket:

```text
B * H_kv * D_bucket * ceil(T / G_bucket) * 2 tensors * bytes_per_meta
```

### Value metadata

For each bucket:

```text
B * H_kv * T * ceil(D_bucket / G_bucket) * 2 tensors * bytes_per_meta
```

### INT2 payload

Use packed INT2 size, not fake tensor size, for paper-facing estimates.

## Acceptance criteria

- Can compare KIVI and CAGE under the same prompt length and residual length.
- Reports both payload-only bytes and total cache bytes.
- Handles fake-quant cache records without claiming fake FP16 tensors are the final compressed representation.
- Output can be serialized to JSON.
