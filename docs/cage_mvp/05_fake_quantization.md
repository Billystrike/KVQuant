# Task 05: INT2 Fake Quantization

## Objective

Implement correctness-first bucketed INT2 fake quantization for Key and Value caches before modifying CUDA pack or matmul kernels.

## Required module

Create `models/cage_quant.py` or add focused helpers to `models/utils_quant.py` if that file already contains the relevant quantization utilities.

## Required interfaces

```python
fake_quant_k_by_channel_buckets(
    key_states,
    bucket_indices,
    group_sizes,
    clip_percentiles,
    bits=2,
) -> torch.Tensor
```

```python
fake_quant_v_by_channel_buckets(
    value_states,
    bucket_indices,
    group_sizes,
    clip_percentiles,
    bits=2,
) -> torch.Tensor
```

## Shape contract

### Key

- input: `[B, H_kv, T, D]`
- select bucket channels from `D`;
- internally quantize selected channels along token dimension `T`;
- scatter fake-quantized values back to `[B, H_kv, T, D]`.

### Value

- input: `[B, H_kv, T, D]`
- select bucket channels from `D`;
- quantize selected channels along channel dimension within each token;
- scatter fake-quantized values back to `[B, H_kv, T, D]`.

## Quantizer definition

Use asymmetric 2-bit quantization with optional percentile clipping:

```text
mn = quantile(x, lower_q)
mx = quantile(x, upper_q)
scale = max(mx - mn, eps) / (2^bits - 1)
code = round(clamp((x - mn) / scale, 0, 2^bits - 1))
x_hat = code * scale + mn
```

For percentile `p`:

```text
lower_q = (1 - p) / 2
upper_q = 1 - lower_q
```

If clipping is disabled later, use min/max.

## Numerical requirements

- clamp scale with a small epsilon;
- avoid NaN/Inf with `torch.nan_to_num`;
- preserve dtype expectations where possible;
- compute quantile in a way that is acceptable for fake-path correctness even if it is not fast.

## Acceptance criteria

- Output shape exactly matches input shape.
- Single-bucket fixed group fake quant can approximate the existing KIVI fixed-group behavior.
- Multi-bucket fake quant changes only selected bucket channels and scatters back correctly.
- No CUDA changes are required.
