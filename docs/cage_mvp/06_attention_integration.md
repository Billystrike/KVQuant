# Task 06: Attention Integration and Short Generation Path

## Objective

Integrate the CAGE-KV fake path into `LlamaAttention_KIVI.forward` so short-prompt generation can run end-to-end with `config.cage_enable=True`.

## Scope

Start with `models/llama_kivi.py`. Mistral should be handled only after Llama proves the interfaces.

## Required behavior

### CAGE disabled

When `config.cage_enable=False`, execution must remain on the original KIVI path.

### Prefill with CAGE enabled

1. Compute query, key, and value states as usual.
2. Compute Key importance using `compute_key_importance`.
3. Compute Value importance using `compute_value_importance`.
4. Assign Key and Value channel buckets.
5. Apply bucketed INT2 fake quantization to historical Key and Value states.
6. Continue attention computation with fake-quantized tensors.
7. Store a CAGE-compatible cache object for decode.

### Decode with CAGE enabled

The MVP may use a slow but correct path:

1. Reuse prefill bucket policy.
2. Append new full-precision token states.
3. Reconstruct or fake-quantize the current cache as needed for correctness.
4. Avoid decode-time bucket reshuffling.

## Required safeguards

- CAGE fake path should raise a clear error if the sequence shape is unsupported.
- Bucket policy must not be recomputed every decode token unless explicitly requested later.
- Cache length must match attention mask expectations.

## Acceptance criteria

- A short prompt can generate 8 to 16 tokens with CAGE enabled.
- Attention output shape matches the original path.
- CAGE disabled output path remains unchanged.
- No packed CUDA CAGE path is required in this task.
