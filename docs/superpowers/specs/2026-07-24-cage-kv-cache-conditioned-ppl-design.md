# CAGE-KV cache-conditioned continuation PPL pilot

## Objective

Measure the language-model quality effect of FP16, KIVI, and CAGE KV-cache
representations while exercising the same incremental cache path used during
autoregressive decoding. This is a native-context diagnostic and screening
experiment. It is not a canonical full-corpus WikiText-2 perplexity result.

## Why a separate metric is required

A single full-sequence forward pass computes all token states together and
does not consume a previously stored quantized KV cache. It therefore cannot
measure the intended cache perturbation. Each case in this pilot performs one
prefill followed by one-token teacher-forced decode steps with the returned
`past_key_values`.

## Frozen corpus identity

- Dataset: `Salesforce/wikitext`
- Configuration: `wikitext-2-raw-v1`
- Split: `test`
- Revision: `00aa25585682d4957f9e86edc73f59be7419af99`
- Join rule: `"\n\n".join(test["text"])`
- Tokenization: Llama-2 tokenizer with `add_special_tokens=False`
- Rows: 4,358, of which 2,891 are non-empty
- Joined UTF-8 bytes: 1,296,370
- Joined text SHA-256: `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`
- Token count: 341,468
- Token-ID SHA-256: `8163e5b39c668be8eec1e4d82eaecb1ee2a09d958f81873773e93bf4a4484f10`

The runner checks every value above before loading model weights. The dataset
library fingerprint is recorded as diagnostic provenance but is not used as a
content identity because it may vary across compatible loader versions.

## Deterministic inputs

Five continuation anchors are selected at the interior sixths of the valid
token range. An anchor must have 4,032 preceding tokens and 64 following
tokens. For anchor index `i` in `[0, 4]`:

```text
minimum = 4032
maximum = token_count - 64
anchor = minimum + floor((maximum - minimum) * (i + 1) / 6)
```

Every anchor uses the same 64 target tokens at all four prompt lengths. The
prompts are nested suffixes of 512, 1,024, 2,048, and 4,032 tokens immediately
preceding the anchor. No BOS/EOS token is inserted into these corpus slices.
The longest case is exactly 4,096 tokens and uses no context extension or RoPE
scaling.

## Scoring

For each case:

1. Prefill exactly `T` prompt tokens with `use_cache=True`.
2. Use the final prefill logit to score continuation token 0. This is the
   boundary target and is reported separately because it is not produced by a
   decode query consuming the stored cache.
3. Teacher-force continuation tokens 0 through 62 one at a time, carrying the
   returned cache, and use their logits to score continuation tokens 1 through
   63.

The primary metric is the aggregate negative log-likelihood and perplexity of
the 63 cache-dependent decode targets. The 64-target result including the
prefill boundary is secondary. NLL is computed from FP32 logits; aggregation
sums NLL and token counts before exponentiation. Per-case token NLL values are
stored for audit and paired analysis.

For FP16 only, a no-cache full-sequence forward pass also scores the same 64
targets. Its difference from incremental scoring is a numerical calibration
diagnostic. A hard tolerance will be chosen only after the real acceptance
run; it is not silently inferred from CPU fakes.

## Experiment matrices

Acceptance uses one anchor, prompt lengths 512 and 4,032, and three methods:

- FP16
- KIVI `(group_size=32, residual_length=32)`
- CAGE `residual_length=32`

This produces six cases.

The full pilot uses all five anchors, all four prompt lengths, and the ten
methods from the memory-perturbation pilot. This produces 200 cases and 12,600
primary cache-dependent target tokens. Acceptance case IDs are a subset of the
full matrix when source state is unchanged.

## Outputs and resume behavior

The output layout is:

```text
manifest.resolved.json
cases/<case_id>.json
failures/<case_id>.json
summary/cases.jsonl
summary/cases.csv
summary/quality.json
```

Case identity includes normalized model identity, resolved method, frozen
corpus content identity, anchor and token-slice hashes, scoring protocol, and
tracked source state. A completed case is reused only after strict schema and
internal-consistency validation. Aggregation is limited to expected IDs from
the current resolved manifest.

## Interpretation boundary

The primary paper-facing quantity is length-stratified cache-conditioned
continuation PPL and paired delta versus FP16. The five deterministic anchors
are a designed diagnostic sample, so dispersion and paired differences do not
establish population-level statistical significance. Overall PPL across prompt
lengths repeats each continuation four times and is diagnostic only.

CAGE remains a fake-quant prototype without a fused variable-group CUDA
kernel. Runtime and CUDA peaks are diagnostics, not compressed-kernel
performance claims. This pilot does not replace standard full-corpus PPL,
LongBench, QA, or other downstream evaluation.
