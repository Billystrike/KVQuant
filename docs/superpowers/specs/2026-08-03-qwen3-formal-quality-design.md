# Qwen3-8B formal matched-memory quality design

## Freeze boundary

This design was frozen after Kitty/Qwen3 acceptance, CAGE and KIVI Qwen3
acceptance, complete packed-memory validation, and tokenizer/corpus auditing,
but before any formal Qwen3 quality result. The machine-readable authority is
`configs/qwen3_8b_formal_quality_protocol_v1.json`. The earlier
`configs/qwen3_8b_matched_memory_protocol_v1.json` remains the authority for
packed-byte eligibility and is not replaced by this file.

The Qwen token audit passed on source commit
`a9468852d5e793e23303e972f23a185082e5fbd5`. The neutral raw-corpus snapshot
SHA-256 is
`f9943f0c88bea153f2eb8e1b04fee9b65fbda2753746eacaed36ecc07718000a`;
the Qwen audit JSON SHA-256 is
`d60df0c9d32ea77c826794a6f7b8ea7804344603024e1759599ec19f62784930`;
and the complete audit log SHA-256 is
`cc03723dff39dd001db086d9fa81e75e869257cdcdb26fe0c2312b77ea569507`.

## Paired inputs and scoring

The source is the frozen WikiText-2 raw test snapshot. Qwen tokenization gives
299,078 tokens with token-ID SHA-256
`069de3e782d0f61736e593f6d995478fcb9ae8377327ecfa1d57dc10045b3a7c`.
Fifty deterministic interior-51sts anchors are used at prompt lengths 1024,
2048, and 4032 with a shared 64-token continuation. The minimum anchor gap is
5,783, greater than the largest 4,096-token prompt-plus-continuation window.

All methods score exactly the same 64 continuation targets. The final prompt
logit scores continuation token zero. Teacher-forced one-token cache updates
then score continuation tokens one through 63. FP16 uses this same incremental
rule; it is not calibrated through a separate one-shot scoring path. This is
cache-conditioned continuation NLL, not canonical full-corpus perplexity.

The tokenizer warning produced while encoding the complete 299,078-token
offline corpus stream is outside model execution. Every model case is at most
4,096 tokens, below the checkpoint's native 40,960-token context.

## Frozen matrix

The unique base matrix contains 18 method-length points per anchor: FP16,
Kitty, and Kitty-Pro at all three lengths; five distinct CAGE points selected
by the packed-memory protocol; and four distinct KIVI points. This yields 900
base cases. A missing KIVI match for the 2048-token Kitty target remains
missing because its closest grid point failed the frozen three-percent gate.

The ten primary CAGE-minus-baseline comparisons are listed explicitly in the
machine protocol. A shared CAGE or KIVI point is executed once even when it
supports both Kitty and Kitty-Pro target comparisons. Negative paired delta
NLL favors CAGE.

## Exact-byte mechanism controls

Mechanism controls are frozen at CAGE r288 with length 2048 and CAGE r96 with
length 4032. Each point adds fixed-random, fixed-uniform, Key-adaptive-only,
and Value-adaptive-only variants. Unlike the earlier Llama uniform one-bucket
diagnostic, these Qwen controls retain all three bucket sizes, group sizes,
clip percentiles, residual storage, metadata, and INT64 index counts. Only the
channel ordering changes, so every variant has exactly the same packed-byte
charge as full CAGE at its point.

The fixed-uniform assignment is deterministic and strided over channel index.
For head dimension 128 and three buckets it maps channel residues so the
resulting bucket sizes remain exactly 42, 43, and 43. Key-adaptive-only uses
Q2-variance ordering for Key and fixed-uniform Value; Value-adaptive-only uses
fixed-uniform Key and output-projection/variance ordering for Value.

These eight additional method-length points per anchor add 400 cases. The
complete formal PPL design therefore contains 1,300 cases and 83,200 scored
targets.

## Execution isolation and integrity

FP16, KIVI, CAGE, and CAGE mechanism variants run in the isolated
`cage-qwen3` environment. Kitty and Kitty-Pro run in `kitty-qwen3`. Environment
separation does not permit input drift: each partition must reconstruct and
verify the frozen corpus, token-stream, prompt, continuation, and full-case
hashes before executing a case.

Outputs are append-only and resumable by the unique tuple `(method_id,
prompt_length, anchor_index)`. Resume is allowed only under exact protocol,
model, source, and dependency identity. Conflicting duplicates, non-finite
numbers, mismatched token hashes, or partial method records fail validation.
A small acceptance matrix must be repeated into a fresh output before the full
run begins.

## Analysis and claims

Each declared contrast uses 50 paired anchor differences. Report their mean,
median, population standard deviation, extrema, direction counts,
`exp(mean delta NLL)`, and a deterministic 10,000-resample percentile
bootstrap interval using seed 20260803. The interval describes stability over
the systematic corpus grid and is not a population-significance claim.

All unfavorable primary results remain in the paper package. Passkey remains
a saturation diagnostic. CAGE and accuracy-simulation results support quality
and packed paper-estimate claims only, not fused-kernel latency, throughput, or
realized CUDA-memory savings.
