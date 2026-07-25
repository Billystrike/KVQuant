# CAGE-KV 50-anchor paired PPL design

## Purpose

The first five-anchor cache-conditioned PPL pilot showed that CAGE and the
strongest KIVI configurations differ by much less than their variation across
corpus locations. This follow-up increases coverage before any further claim
selection. It tests two comparisons declared in advance:

1. `cage-r128` versus `kivi-g32-r128`: equal residual length and the
   highest-fidelity KIVI grouping in the pilot.
2. `cage-r64` versus `kivi-g64-r64`: equal residual length and a
   memory-efficient KIVI operating point.

Paired difference is always candidate CAGE mean NLL minus baseline KIVI mean
NLL. Negative values favor CAGE.

## Frozen matrix

The corpus, tokenizer identity, native context, 64-token continuation, and
63 cache-dependent primary targets are unchanged from the first PPL pilot.
Methods are:

- FP16
- KIVI g32-r128
- KIVI g64-r64
- CAGE r64
- CAGE r128

Prompt lengths are 512, 1024, 2048, and 4032. Fifty deterministic continuation
starts form the interior 51sts grid:

```text
start(i) = 4032 + floor((token_count - 64 - 4032) * (i + 1) / 51)
i = 0, ..., 49
```

Adjacent starts are more than 4032 tokens apart in the frozen 341,468-token
corpus. Therefore even the maximum-context prompt-plus-continuation windows do
not overlap. The full matrix contains `5 × 4 × 50 = 1000` cases and 63,000
primary targets. The ten-case acceptance matrix uses anchor 0 and lengths 512
and 4032; its IDs are a subset of the full matrix at unchanged source state.

## Pre-registered analysis

Each prompt-length comparison contains 50 paired anchor differences. The
overall diagnostic first averages the four prompt-length differences within
each anchor, leaving 50 anchor clusters rather than treating 200 repeated
contexts as independent.

For each declared comparison and each length, report:

- mean and median paired delta NLL;
- population standard deviation, minimum, and maximum;
- counts of anchors favoring CAGE, tied, or favoring KIVI;
- `exp(mean delta NLL)`, interpreted as the CAGE/KIVI PPL ratio;
- a deterministic percentile bootstrap interval for mean delta NLL.

The bootstrap uses 10,000 resamples and base seed 20260725. One anchor is the
resampling unit. The 2.5th and 97.5th percentiles are descriptive stability
intervals over this systematic corpus grid. The anchors are not random
population draws, so these intervals are not population-level significance
claims.

FP16 incremental-versus-one-shot calibration retains the previously frozen
limits: maximum per-case mean absolute token-NLL delta at most 0.005 and
maximum individual token-NLL delta at most 0.03.

## Interpretation boundary

This experiment can strengthen or weaken the claim that CAGE preserves
cache-conditioned language-model quality relative to selected KIVI operating
points. It does not measure canonical full-corpus perplexity, realized packed
memory, latency, or throughput. CAGE remains a fake-quant prototype without a
fused variable-group kernel.
