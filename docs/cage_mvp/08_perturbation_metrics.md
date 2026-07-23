# Task 08: Perturbation Metrics

## Objective

Add internal metrics that demonstrate why CAGE-KV improves functional error, not just reconstruction error.

## Required module

Create `utils/cage_metrics.py`.

## Key metrics

- `relative_k_reconstruction_error`
- `attention_logit_mse`
- `attention_score_kl`
- `topk_attention_overlap`
- `weighted_key_error = sum(I^K_c * mse(K_c))`

## Value metrics

- `relative_v_reconstruction_error`
- `attention_output_mse = ||A V - A V_hat||^2`
- `post_o_proj_mse = ||(A V - A V_hat) W_O||^2`
- `weighted_value_error = sum(I^V_c * mse(V_c))`

## Logging contract

Metrics should be returned as plain Python floats in a dictionary that can be dumped as JSON or JSONL.

## MVP simplifications

- Compute metrics on a single batch or a small debug batch.
- It is acceptable to compute metrics only during prefill.
- Metrics should be optional and gated by `config.cage_collect_metrics`.

## Acceptance criteria

- Metrics can be computed for FP16 vs KIVI and FP16 vs CAGE on the same tensors.
- Metric collection can be disabled with zero overhead except simple branch checks.
- JSONL logging works when `config.cage_dump_dir` is provided.
