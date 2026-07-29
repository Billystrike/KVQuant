# CAGE-KV Llama-2-7B final paper package design

## Purpose

This package closes the Llama-2-7B experimental evidence chain without
creating new measurements. It joins three already frozen analysis packages:

1. the memory--perturbation, paired-PPL, and passkey paper-evidence ledger;
2. the 144-run local mechanism ablation;
3. the 2,000-case paired PPL mechanism ablation.

The earlier evidence ledger remains immutable. The final package consumes it
as an input so that the source analyses and their original protocol records
remain independently reproducible.

## Join policy

Method roles and residual lengths must match exactly. Local perturbation and
PPL mechanism rows join only at prompt lengths 512, 1024, and 2048. Local
length 4095 and PPL length 4032 are retained as two distinct unmatched rows;
they are never silently treated as equal.

The primary within-CAGE mechanism contrasts are full CAGE versus:

- the same-budget fixed-random assignment control;
- uniform group size 64;
- Key-adaptive/Value-uniform;
- Key-uniform/Value-adaptive.

Negative full-minus-baseline error or NLL favors full CAGE. For readability,
the paper tables also report positive percentage improvement where positive
means that full CAGE is better.

## Claim policy

The package separates supported, mixed, and bounded claims. In particular:

- a bootstrap interval excluding zero over deterministic anchors is reported
  as descriptive stability, not population-level statistical significance;
- failure to detect a difference is not called equivalence;
- local reconstruction evidence and cache-conditioned PPL evidence are kept
  distinct;
- saturated passkey accuracy is not presented as a quality improvement;
- paper-estimate packed bytes are not runtime allocation, CUDA peak, latency,
  throughput, or fused-kernel performance.

## Outputs

The package writes six auditable tables in CSV and JSONL form, a protocol,
claim ledger, paper-results draft, LaTeX tables, and a cross-metric mechanism
figure in PNG and PDF form. All outputs are derived read-only from frozen
analysis directories.
