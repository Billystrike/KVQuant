# CAGE-KV MVP Task Documents

This directory decomposes the first CAGE-KV minimum viable deliverable into implementation-ready task documents. The goal is to guide future code changes without mixing design decisions into scattered comments.

The MVP scope is intentionally correctness-first:

- `config.cage_enable=True` gates all new behavior.
- Key importance uses `E[q^2] * Var(K)`.
- Value importance uses `||W_O||^2 * Var(V)` as the default no-attention fallback.
- Channels are assigned to three buckets: high, middle, and low importance.
- Historical KV payload remains all INT2 in the fake-quant prototype.
- Short-prompt generation must run end-to-end.
- Memory summary and perturbation metrics must be emitted for comparison with KIVI.

## Task map

1. [Configuration and feature gate](./01_config_and_feature_gate.md)
2. [CAGE cache structure](./02_cache_structure.md)
3. [Key and Value importance](./03_importance_estimators.md)
4. [Bucket assignment policy](./04_bucket_policy.md)
5. [INT2 fake quantization](./05_fake_quantization.md)
6. [Attention integration and generation smoke path](./06_attention_integration.md)
7. [Memory accounting](./07_memory_summary.md)
8. [Perturbation metrics](./08_perturbation_metrics.md)
9. [MVP tests and smoke scripts](./09_tests_and_smoke.md)

## Non-goals for the MVP

- No fused CUDA variable-group kernel.
- No Kitty-style INT4 boosted channel payload.
- No decode-time bucket reshuffling.
- No requirement to optimize latency in the first pass.
- No requirement to support every model family before the Llama path works.

## Implementation order

Follow the task map order. Each document defines its own scope, required interfaces, acceptance criteria, and risks. Later tasks should depend only on public helpers introduced by earlier tasks.
