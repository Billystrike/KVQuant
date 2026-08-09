# Qwen3 CAGE-v2 feasibility boundary

This development track is separate from the frozen CAGE-v1 Qwen3 experiment.
No v1 protocol, result receipt, acceptance gate, or result archive is replaced.

## Representation

CAGE-v2 keeps an all-channel INT2 Key/Value backbone.  For Key only, the
query-aware `q2_var` score selects two disjoint sparse residual streams:

- selected one-bit channels store `Q1(K - dequant(Q2(K)))`;
- selected two-bit channels store `Q2(K - dequant(Q2(K)))`;
- Value uses uniform per-token INT2 and no channel adaptation.

The selected channel indices are prompt-persistent.  Their payload, scale,
zero-point, and index bytes are all included in the logical packed estimate.
The accuracy path remains fake quantization and supports no realized CUDA
memory, latency, or throughput claim.

## Budget allocation

Per-layer one-bit and two-bit channel quotas are explicit configuration values.
`utils/qwen3_cage_v2.py` provides a deterministic multiple-choice allocator for
calibrated layer options.  It minimizes expected distortion subject to an exact
model-wide byte ceiling and reports unused bytes; quality is not allowed to
change the byte target after selection.

## Data separation

The prior WikiText-2 test 50-anchor grid has already been observed and is not a
valid final holdout for CAGE-v2.  The draft feasibility protocol uses ten
non-overlapping anchors from the WikiText-2 validation split and marks every
derived artifact as development-only.  A distinct final holdout must be chosen
and frozen after development and before final GPU results.

## Development limit

At most two design rounds and three candidate families per round are permitted.
Local perturbation is screened before paired continuation NLL.  The development
gate requires exact packed-byte compliance, improvement over CAGE-v1 at every
length, and no worse performance than matched Kitty at least two of three
lengths.  Passing this gate permits a frozen final protocol; it is not itself a
paper claim.
