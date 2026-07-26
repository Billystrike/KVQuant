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

## Post-run deviation record (2026-07-26)

This section was added after the frozen 1,000-case matrix completed and must
not be read as part of the pre-registration. The per-case mean calibration
gate passed (`0.0032123428350701033 <= 0.005`), while the maximum individual
token gate failed (`0.030837535858154297 > 0.03`). Three primary decode tokens
in three cases exceeded the limit out of 12,800 FP16 calibration comparisons;
the maximum excess was `0.000837535858154298`. The frozen threshold is not
retrospectively raised.

The raw matrix is retained. Primary analysis is restricted to the two
pre-registered incremental CAGE-versus-KIVI comparisons, which do not consume
the FP16 one-shot reference. FP16 remains a diagnostic reference and the
calibration outcome is reported as `FAIL` with protocol status
`RECORDED_DEVIATION`. Canonical full-corpus PPL and exact incremental/one-shot
equivalence remain outside the claim scope.

The implementation plan called for a separate-output acceptance repeat before
the full run. That repeat was inadvertently deferred until after full-matrix
completion. It is therefore a post-full reproducibility audit, not evidence
that the original execution order conformed to the plan. Its timing and result
must be retained with the archived artifacts. The completed audit compared 985
scientific numeric fields, all 985 were bitwise identical, and the worst delta
was zero. Its archive SHA-256 is
`64d4a3ac1a2c3520691f68e620f474e1747a594df54d4036f2baff4c647e8ac7`.

## Interpretation boundary

This experiment can strengthen or weaken the claim that CAGE preserves
cache-conditioned language-model quality relative to selected KIVI operating
points. It does not measure canonical full-corpus perplexity, realized packed
memory, latency, or throughput. CAGE remains a fake-quant prototype without a
fused variable-group kernel.
