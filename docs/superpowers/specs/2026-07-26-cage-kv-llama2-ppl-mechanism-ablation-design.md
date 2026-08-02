# CAGE-KV Llama-2-7B paired PPL mechanism ablation design

## Purpose

The frozen local mechanism study shows that importance ordering improves the
joint post-output-projection MSE over a same-budget fixed-random assignment in
all 24 sample-level comparisons. Key-only and Value-only variants each beat
the uniform control at all four prompt lengths for residual lengths 64 and
128. This follow-up tests whether those local mechanism effects transfer to
cache-dependent continuation NLL.

## Frozen methods

At each residual length in `{64, 128}`, evaluate:

1. full adaptive Key and Value;
2. adaptive Key with uniform Value;
3. uniform Key with adaptive Value;
4. uniform Key and Value;
5. fixed-random Key and Value with the full three-bucket schedules.

All method configurations, bucket schedules, clip percentiles, and the
fixed-random base seed 1729 are identical to the frozen local ablation. No
method is selected or removed using PPL outcomes. FP16 and KIVI are not rerun:
this experiment estimates within-CAGE mechanism contrasts, while the frozen
50-anchor paired study already supplies the direct full-CAGE versus KIVI
comparison.

## Frozen inputs and scale

- model: Llama-2-7B, native 4096 context, no RoPE scaling;
- corpus: the existing immutable WikiText-2 raw test token snapshot;
- prompt lengths: `512, 1024, 2048, 4032`;
- continuation: 64 tokens, with 63 cache-dependent decode targets;
- selection: `interior-51sts-v1`;
- anchors: the same deterministic 50 anchors used by the direct paired PPL
  study.

The full matrix is `10 methods x 4 lengths x 50 anchors = 2,000 cases`, with
126,000 primary cache-dependent targets. Acceptance is all 10 methods at
lengths 512 and 4032 on anchor 0: 20 cases. Acceptance is a strict subset of
the full matrix and shares its output directory.

## Pre-registered comparisons

For each residual length, the primary contrasts are:

- full minus fixed-random;
- full minus uniform;
- full minus Key-only adaptive;
- full minus Value-only adaptive.

Negative paired delta NLL favors full CAGE. Secondary decomposition reports
Key-only minus uniform, Value-only minus uniform, and the factorial interaction
`full - key_only - value_only + uniform`. The same comparisons are reported
overall and separately by prompt length. The overall unit first averages the
four lengths within each anchor.

## Descriptive uncertainty and boundaries

Use deterministic paired bootstrap intervals over the 50 anchor clusters with
10,000 resamples and seed 20260726. These intervals describe stability over
the frozen corpus grid; they are not population-significance claims. Preserve
all adverse and favorable contrasts. Do not interpret runtime diagnostics as
kernel speed, and do not convert this experiment into a LongBench or general
downstream-quality claim.
