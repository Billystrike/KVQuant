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

The neutral manifest gate passed on source commit
`f6a182a890eb4dc36f5de8e2a2d118a22c744ad1`. It produced exactly 150 ordered
inputs from `qwen3-a00-l1024` through `qwen3-a49-l4032`; its SHA-256 is
`911728ebd24683520c1378520ef02a0669aecfddf03fb4a458c0b73e280e07c5`.
Both isolated Conda environments independently validated that same file. The
complete gate log SHA-256 is
`527d5c12c579b4f980d4be6e65f1604246b87a5764fb32daa8ae038494510f95`.

## Execution gate

The separate formal execution config freezes 20 CAGE-partition method-length
points (1,000 full cases) and six Kitty-partition points (300 full cases). Its
acceptance subset contains 11 CAGE-partition cases and three Kitty-partition
cases, all at anchor zero. Each partition must be executed twice into fresh
output directories and its scientific fields must be bitwise identical. The
execution-config SHA-256 is
`c7a1634613cfeeecd604d2f056ea77a0f9371d42a1a0fc6fc199c27dffa26f5c`.

The initial CAGE-partition runner is deliberately locked against `full` stage
execution. It can create only acceptance outputs until both repeated
partition gates have passed and their hashes have been recorded. Progress and
completion summaries omit NLL values so the acceptance run cannot be used to
modify the frozen samples, methods, budgets, or analysis.
