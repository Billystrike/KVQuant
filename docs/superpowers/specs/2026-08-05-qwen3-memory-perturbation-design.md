# Qwen3-8B Unified Memory--Perturbation Design

## Status and role

This protocol was frozen after the 1,300-case formal Qwen3 quality analysis and
before any Qwen3 perturbation result was executed. It is therefore a disclosed
post-quality local-mechanism validation, not an input to method selection or a
replacement for the paired continuation-NLL results.

The executable JSON specification is
`configs/qwen3_8b_memory_perturbation_protocol_v1.json`.

## Grid

The experiment inherits every method-length point and every one of the 50
anchors from the already frozen formal quality experiment:

- 20 FP16/CAGE/KIVI method-length points, or 1,000 cases;
- 6 Kitty/Kitty-Pro method-length points, or 300 cases;
- 36 decoder-layer records per case;
- 1,300 cases and 46,800 layer records in total.

No point is removed or added using the observed quality results. Packed memory
is recomputed at the prompt length with the frozen matched-memory estimator.

## Controlled local measurement

For each case, the runner performs three phases:

1. An FP16 no-cache pass over the prompt plus continuation token 0 captures the
   final-position query at each layer.
2. Candidate prefill over the prompt captures the unquantized candidate prefix
   K/V and the Key and Value importance weights while constructing the actual
   method cache.
3. A one-token teacher-forced candidate decode consumes continuation token 0.
   At every layer, the exact history is the captured unquantized prefix plus
   that candidate step's current-token K/V. The perturbed history is the K/V
   returned by the method cache to the same attention computation.

Both histories use the captured FP16 final-position query. Candidate-path
queries are excluded so upstream errors are not mixed into the controlled local
cache perturbation. End-to-end propagation remains measured only by the frozen
paired quality experiment.

This is the same causal definition used by the frozen Llama pilot. In
particular, it does not compare a quantized prefill path with another method's
unperturbed prefill output.

## Metrics and integrity

The primary local functional axis is `joint_post_o_proj_mse`. The complete
12-metric Llama schema is retained, including reconstruction errors,
attention-logit and score changes, top-k overlap, Value-only output changes,
and joint output changes. Each case records all 36 layers and canonical mean,
median, and maximum aggregates.

FP16 must produce exactly zero for all error metrics and exactly one for top-k
overlap. Quantized methods must perturb both Key and Value storage. All cache
tensors, source identities, model identities, input hashes, lengths, and packed
memory reports are validated before a case becomes resumable.

## Isolation and gates

One runner emits a common schema but is invoked separately:

- `cage_qwen3` in `/root/autodl-tmp/conda-envs/cage-qwen3`;
- `kitty_qwen3` in `/root/autodl-tmp/conda-envs/kitty-qwen3`.

The runner was initially acceptance-only. Two fresh acceptance outputs per
partition were required to be bitwise-consistent in the frozen scientific
payload. Both pairs passed, and their artifact identities are frozen in
`configs/qwen3_8b_memory_perturbation_acceptance_gate_v1.json`. Full execution
is unlocked only when the checked-in gate and every server-side acceptance
artifact are revalidated. This prevents an unvalidated instrumentation path
from producing formal results.

## Claim boundary

The experiment supports local mechanism and packed paper-estimate
memory--perturbation statements. It does not establish fused-kernel latency,
throughput, realized CUDA-memory savings, population inference, or end-to-end
quality beyond the already frozen continuation-NLL experiment.
