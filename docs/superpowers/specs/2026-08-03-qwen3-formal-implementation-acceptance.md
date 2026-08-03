# Qwen3 formal-quality implementation acceptance record

This record is downstream of the frozen formal-quality protocol and does not
change its samples, methods, budgets, endpoints, or analysis. No formal Qwen3
quality result had been executed when this record was created.

## Exact-byte fixed-uniform control

The implementation was accepted on commit
`04817e46b96ed076ea21466e06579f77fdd79397` with the frozen protocol SHA-256
`78f33126ace6822649c623b6a2535abc23262728e51818d74b0d14b92fe10e92`.
All 51 CPU and protocol tests passed. The Qwen3-8B GPU acceptance report had
`status=pass`, no failures, exact prefill logits, finite generation and resume,
and exact Key and Value fixed-uniform indices in all 36 layers. Each side used
the frozen `(8,42)`, `(8,43)`, `(8,43)` bucket shapes with INT64 indices.

The GPU acceptance JSON SHA-256 is
`795e6a0a1415c66755c22b42e602d841959b0a75f1367dc223761298caf26337`.
The complete log SHA-256 is
`c1ba20e9d1aec8254c98dfdffbd36d8fda00d8f54d62a2253b793fd9053ef6df`.
The observed peak CUDA allocation was 17,628,246,016 bytes and is an
acceptance diagnostic only, not a packed-cache or realized-memory result.

## Neutral input-manifest gate

Before either execution environment may run a quality case, a single neutral
input manifest must be built from the frozen raw-corpus snapshot and Qwen
tokenizer. It contains exactly 150 unique inputs: 50 anchors at each of 1024,
2048, and 4032 prompt tokens, each sharing its anchor's 64-token continuation.
Every record stores exact prompt, continuation, and combined token hashes.

The CAGE/KIVI/FP16 and Kitty partitions must consume the same manifest file
and verify its complete SHA-256, protocol identity, clean source commit,
overall token-stream identity, case order, token counts, and per-case hashes.
The manifest is an input artifact and contains no model output or quality
measurement.
