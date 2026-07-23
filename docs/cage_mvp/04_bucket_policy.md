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

- `bucket_indices`: tuple/list of per-head index tensors, ordered high to low importance.
  Each bucket tensor has shape `[H, D_bucket]`, because different heads may rank
  different channels as important.
- `bucket_rank_map`: `[H, D]` debug tensor whose value is the assigned bucket id
  for each head/channel pair.
- `num_buckets`: the effective bucket count after capping by `D`.
- optional `valid_channel_counts` when padding is used.

## MVP policy

Use quantile assignment:

- bucket 0: highest importance, smallest group size;
- bucket 1: middle importance;
- bucket 2: lowest importance, largest group size.

The default group-size mapping is provided by config, not hardcoded here.

For the MVP, bucket sizes should be as even as possible while keeping the
highest-importance bucket no larger than lower-importance buckets. For example,
`D=5` and `num_buckets=3` produces bucket sizes `[1, 2, 2]`. This matches the
intended mapping where bucket 0 later receives the smallest quantization group
size and bucket 2 receives the largest group size.

## Determinism requirements

1. Ties must be broken deterministically by channel index.
2. Every channel must be assigned exactly once per head.
3. Empty buckets should be avoided when `D >= num_buckets`.
4. If `D < num_buckets`, reduce the effective bucket count or merge empty buckets gracefully.

## Padding guidance

The fake-quant MVP does not need channel padding. The later packed INT2 path should use `pad_to_multiple=16` or a multiple compatible with group size and pack factor.

When padding is requested, padded entries may repeat a valid channel index to
avoid invalid gathers. Callers must use `valid_channel_counts` to ignore padded
positions when interpreting the assignment.

## Acceptance criteria

- High-importance channels are assigned to bucket 0.
- All channels are covered once and only once.
- The output is stable across repeated runs on identical inputs.
- Bucket assignment can be reused for both Key and Value quantization.
