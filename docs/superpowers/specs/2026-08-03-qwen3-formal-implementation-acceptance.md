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

## CAGE-partition repeated acceptance result

Both fresh 11-case CAGE-partition acceptance runs completed with no failures
on commit `efffe38bf9ea2eeac15d36a13edb51d351ad7ed4`. Their run-identity SHA-256
was identical:
`01e152b627ab59f1067bba9c27bcedc4c4fc76d2d2b4370f058079e5282aab05`.
The A and B archive SHA-256 values are, respectively,
`ce758b423a3085028a213b1541e94dae79dbf1b30c40abfb300349fd903a8a02`
and
`6546e1972cb0e1aa484b785da911e761e68bce36c26c674ea147622d27bc4ace`.

All 11 scientific payloads, including method/input identity, all 64 token NLL
values, aggregate scoring, and cache diagnostics, were bitwise equal. The
shared scientific-payload SHA-256 is
`f4e809d154a120e1375b7ab32b393d86950194595e398a9331fa3d1b93fc7d86`;
the comparison-report SHA-256 is
`7ba35a4cf603545fe65ced2d78bdc67521bdd415d6f8c3fd0b8dee9c56921857`.
The A log SHA-256 is
`ced5656a04620c429045adebce01abc797ec3bc4b6b5faf10cebe99896ef1089`;
the B-and-comparison log SHA-256 is
`379e179b4a3baeb253ef22f82d923075c9756f23890a27d14c8a832a568ec0f6`.

This closes the CAGE-partition acceptance gate but does not unlock either full
partition. Kitty must independently pass its two fresh acceptance runs before
the full-stage lock can be reconsidered.

## Kitty pre-case identity-gate correction

The first Kitty acceptance-A invocation on CAGE commit
`7520d53f50c4d2b0c5733e6d4bcd475cc7d83d19` stopped at the model/source
identity gate before executing any case. Its output directory contained only
`run_identity.json`; no input was scored and no quality result was produced.
The runner had incorrectly associated Git blob
`1149778f674ba49aa2dcb5d6d7b62fcff3e29fa9`, which belongs to
`src/kitty_sim/eval/eval_helper.py`, with `src/kitty_sim/kitty_simulate.py`.

The frozen commit tree, clean worktree, and content read directly from commit
`dfd2c07b407d6b407179359207c612ab631f3ed1` all independently identify the
`kitty_simulate.py` blob as
`1a6dd10496a6692cc34ea340826339e354d4f7fd`. Its SHA-256 is
`c63d8bff512594a7bd5412b8c201e8a706b2757abbfabd8a95436b9f343ee133`.
The already-correct `utils_quant.py` blob and SHA-256 are
`907fd6b1951b869c33216d8eb658df21e8bf9c29` and
`6bb6df3bbfb702793cff79d712f908f7dee89ada66d3bffc399718b218bfc27a`.
The corrected runner verifies both Git blob identity and SHA-256 identity for
both files. A new acceptance A must use a fresh output directory keyed by the
correcting CAGE commit; the failed directory is retained as audit evidence.

## Kitty-partition repeated acceptance result

Both fresh three-case Kitty-partition acceptance runs completed with no
failures on CAGE commit `40dcc8a9b19d64c719a08735ee2e32c1f221c93e`,
Kitty commit `dfd2c07b407d6b407179359207c612ab631f3ed1`, and Transformers
commit `37f8b0b53512e6aae0cfd15746c133c101783178`. Their common run-identity
SHA-256 is
`4bc505acbaf34f32072a6342397c632ce74ec11daa5fa8b1daeabfaaa12dfd85`.

All three scientific payloads were bitwise equal. Their shared payload SHA-256
is `96617030ab09193ad4381b8b41439b5fec4433fd1e1ab269a9a241c2671b83a3`,
and the comparison-report SHA-256 is
`7ec8f021cff7fa3e1e34cc521e2d9517877cf5f3bc73e0f8abb058d5d8baa7f8`.
The A and B archive SHA-256 values are
`b5fb13b19b047774818d954846d5ce56a1cc59b8dc9d0009bdd57e3413cb519d`
and `48407f4454f0f1269ac4fa759120f5a779badc198e1d690bb12a83dbf2b862ab`;
the corresponding log SHA-256 values are
`8a462312ee24b19bcb137b7959041bca64116b183e15a86103796f1fae18cebc`
and `565329fee449199c9fb91e08351d5deeed2776823340d5d62d684d4530d04cf9`.

Both runs observed the exact official accuracy-simulation mechanics: 288,
576, and 1116 Key fake-quant calls at prompt lengths 1024, 2048, and 4032;
2304 Value fake-quant calls per case; and respectively 16, 32, and 32 promoted
channels per head. All cache diagnostic checks passed.

## Full-execution gate

The CAGE and Kitty repeated-acceptance evidence closes both partition gates.
`configs/qwen3_8b_formal_acceptance_gate_v1.json` records the immutable
artifact paths and hashes. Full runners require this gate explicitly and
re-hash both run locks, both summaries, both archives, both logs, both
comparison reports, and the retained zero-case failed-attempt evidence before
expanding a full matrix. The gate SHA-256 is embedded into every full run and
case identity. The pre-result protocol, execution config, and input manifest
remain unchanged.
