# CAGE-KV paired PPL post-run deviation record

## Status

This record was created on 2026-07-26 after inspecting the completed frozen
matrix produced from source commit
`d1a6ae26fb79124a75a181c5aaf191cc27f84a42`. It is a post-run amendment, not a
pre-registration.

## Frozen observation

- completed cases: 1,000/1,000;
- failed case artifacts: 0;
- FP16 calibration cases: 200;
- FP16 calibration token comparisons: 12,800;
- maximum per-case mean absolute token-NLL delta:
  `0.0032123428350701033` against limit `0.005` (`PASS`);
- maximum individual absolute token-NLL delta:
  `0.030837535858154297` against limit `0.03` (`FAIL`);
- individual-token violations: 3 in 3 cases, all primary decode targets;
- maximum excess over the frozen limit: `0.000837535858154298`.

The three violations occurred at distinct prompt-length/anchor pairs. The
frozen limit is not raised or redefined after observing these values.

## Analysis decision

The matrix remains valid for the two comparisons declared before execution:

1. CAGE r128 minus KIVI g32-r128;
2. CAGE r64 minus KIVI g64-r64.

Both sides of each comparison use the same incremental cache-dependent scoring
path. The FP16 one-shot reference is not an input to either paired delta.
Accordingly, the primary analysis scope is restricted to those comparisons.
FP16 aggregates remain diagnostic and must not support claims of canonical
full-corpus perplexity or exact incremental/one-shot equivalence.

Analysis outputs must state:

```text
FP16_CALIBRATION_GATE=FAIL
PROTOCOL_STATUS=RECORDED_DEVIATION
PRIMARY_ANALYSIS_SCOPE=pre_registered_cage_vs_kivi_paired_comparisons_only
```

## Execution-order deviation

The implementation plan required a separate-output acceptance repeat before
the full matrix. It was not run at that point. A repeat performed afterward is
retained only as a post-full reproducibility audit. The paper and artifact
record must not imply that the original ordering requirement was satisfied.

The post-full repeat completed 10/10 cases without failures. Its ten run IDs
matched the corresponding cases in the frozen full matrix, and all 985
scientific numeric comparisons were bitwise identical (`worst delta = 0`). The
repeat-audit archive SHA-256 is
`64d4a3ac1a2c3520691f68e620f474e1747a594df54d4036f2baff4c647e8ac7`.
