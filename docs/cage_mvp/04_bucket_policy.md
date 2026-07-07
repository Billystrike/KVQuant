# Task 04: Bucket Assignment Policy

## Objective

Assign channels to high, middle, and low CAGE buckets using deterministic channel importance.

## Required functionality

Implement in `models/cage_importance.py` or `models/cage_policy.py`:

```python
assign_channel_buckets(
    importance,
    num_buckets=3,
    stable=True,
    pad_to_multiple=None,
) -> CageBucketAssignment
```

## Input and output

### Input

- `importance`: `[H, D]` or `[B, H, D]`.

If batch dimension exists, reduce it before assigning buckets unless a later caller explicitly requests per-batch policies.

### Output

A bucket assignment object containing:

- `bucket_indices`: list of index tensors, ordered high to low importance.
- `bucket_rank_map`: `[H, D]` or equivalent debug tensor.
- `num_buckets`.
- optional `valid_channel_counts` when padding is used.

## MVP policy

Use quantile assignment:

- bucket 0: highest importance, smallest group size;
- bucket 1: middle importance;
- bucket 2: lowest importance, largest group size.

The default group-size mapping is provided by config, not hardcoded here.

## Determinism requirements

1. Ties must be broken deterministically by channel index.
2. Every channel must be assigned exactly once per head.
3. Empty buckets should be avoided when `D >= num_buckets`.
4. If `D < num_buckets`, reduce the effective bucket count or merge empty buckets gracefully.

## Padding guidance

The fake-quant MVP does not need channel padding. The later packed INT2 path should use `pad_to_multiple=16` or a multiple compatible with group size and pack factor.

## Acceptance criteria

- High-importance channels are assigned to bucket 0.
- All channels are covered once and only once.
- The output is stable across repeated runs on identical inputs.
- Bucket assignment can be reused for both Key and Value quantization.
