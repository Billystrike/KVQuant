# Task 03: Key and Value Importance Estimators

## Objective

Implement functional channel-importance estimators for the CAGE-KV MVP.

## Required module

Create `models/cage_importance.py`.

## Key importance

Use:

```text
I^K_c = E[q_c^2] * Var(K_c)
```

### Interface

```python
compute_key_importance(
    query_states,
    key_states,
    num_key_value_groups=1,
    reduce_batch=True,
) -> torch.Tensor
```

### Shape contract

- `query_states`: `[B, H_q, T, D]`
- `key_states`: `[B, H_kv, T, D]`
- output with `reduce_batch=True`: `[H_kv, D]`
- output with `reduce_batch=False`: `[B, H_kv, D]`

### GQA handling

If `H_q > H_kv`, query heads must be grouped by KV head. Aggregate `E[q^2]` across the query heads that share one KV head.

## Value importance

Use MVP fallback:

```text
I^V_c = ||W_O[:, c]||_2^2 * Var(V_c)
```

The attention sparsity factor can be added later, but the interface should allow it.

### Interface

```python
compute_value_importance(
    value_states,
    o_proj_weight,
    num_heads,
    num_key_value_heads,
    head_dim,
    attn_weights=None,
    reduce_batch=True,
) -> torch.Tensor
```

### Shape contract

- `value_states`: `[B, H_kv, T, D]`
- `o_proj_weight`: `[hidden_size, H_q * D]`
- output with `reduce_batch=True`: `[H_kv, D]`
- output with `reduce_batch=False`: `[B, H_kv, D]`

## Numerical safety

Both estimators must:

- use `torch.nan_to_num`;
- clamp negative values to zero;
- return zeros for zero-variance channels;
- avoid unnecessary materialization of large temporary tensors.

## Acceptance criteria

- Works for MHA where `H_q == H_kv`.
- Works for GQA shape arithmetic even if the first integration target still asserts MHA.
- Produces deterministic non-negative tensors.
- Does not require `output_attentions=True`.
