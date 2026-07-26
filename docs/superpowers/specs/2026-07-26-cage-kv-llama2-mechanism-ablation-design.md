# CAGE-KV Llama-2-7B mechanism ablation design

## Purpose

The frozen Llama-2-7B evidence does not show unconditional dominance over
KIVI. CAGE r128 saves 1.84--5.52% paper-estimate memory against KIVI g32-r128
with descriptive PPL parity, while CAGE r64 uses 11.14--11.56% more memory than
KIVI g64-r64 but improves the paired PPL ratio by 0.785% overall. The mechanism
study therefore asks which part of the adaptive channel policy changes the
memory--perturbation tradeoff; it does not assume that CAGE wins every axis.

## Frozen questions

1. Does importance ordering outperform a same-budget fixed-random assignment?
2. Is the adaptive Key policy sufficient, is the adaptive Value policy
   sufficient, or are both required?
3. Are the conclusions consistent at residual lengths 64 and 128?

"Key-only adaptive" and "Value-only adaptive" do not disable quantization of
the other cache side. Both Key and Value remain 2-bit fake-quantized. The
non-adaptive side receives the uniform three-bucket schedule.

## Factors

The adaptive side policy is:

- importance ordering: Key `q2_var`, Value `wo_var`;
- group schedule: `[32, 64, 128]` from high to low importance;
- clip schedule: `[0.999, 0.995, 0.99]`.

The uniform side policy is three buckets with group schedule `[64, 64, 64]`
and clip schedule `[0.995, 0.995, 0.995]`. It retains identical bucket-index
structure while making assignment order irrelevant.

The fixed-random control retains both adaptive schedules but replaces Key and
Value ordering with deterministic random scores. Its base seed is 1729. For
layer `l`, Key uses seed `1729 + 2*l` and Value uses `1730 + 2*l`. Thus its
paper-estimate memory must be exactly equal to full CAGE at the same residual
length.

## Frozen matrix

At each residual length in `{64, 128}`, run:

1. full adaptive Key + Value;
2. adaptive Key + uniform Value;
3. uniform Key + adaptive Value;
4. uniform Key + uniform Value;
5. fixed-random Key + Value with the full adaptive schedules.

Include KIVI g64-r64 and KIVI g32-r128 as external operating-point baselines.
The full local matrix is 12 methods x 3 natural-text samples x 4 prompt lengths
`{512, 1024, 2048, 4095}` = 144 points and 4,608 layer records. The acceptance
subset is the same 12 methods on doc-001 at lengths 512 and 2048 = 24 points.

## Primary readouts

- paper-facing packed cache bytes;
- joint post-output-projection MSE;
- Key and Value reconstruction metrics already frozen by the core schema;
- per-sample paired differences between full CAGE and each ablation.

Primary mechanism contrasts are full minus fixed-random, full minus uniform,
full minus Key-adaptive-only, and full minus Value-adaptive-only. Negative
error differences favor full CAGE. No latency, throughput, realized-memory,
or fused-kernel claim is permitted.

## Decision rule for later PPL ablation

Do not preselect PPL ablations by whichever local result looks best. Advance
full, fixed-random, and uniform at both residual lengths. Add a side-only
variant only if its local paired error is not worse than uniform at at least
three of four lengths. The later PPL protocol and anchor count must be frozen
before execution.
