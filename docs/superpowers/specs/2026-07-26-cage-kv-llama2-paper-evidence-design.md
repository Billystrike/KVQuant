# CAGE-KV Llama-2-7B paper evidence synthesis design

## Purpose

This synthesis connects the frozen memory–perturbation Pareto pilot, the
50-anchor paired-PPL study, and Passkey Stage B before choosing mechanism
ablations. It does not create new GPU measurements or replace the individual
analysis protocols.

The two declared operating-point comparisons are:

1. CAGE r128 versus KIVI g32-r128;
2. CAGE r64 versus KIVI g64-r64.

## Exact join policy

Method IDs and prompt lengths must match exactly. Memory–perturbation lengths
512, 1024, and 2048 join the same paired-PPL lengths. Pareto length 4095 and
paired-PPL length 4032 remain separate unmatched native-context settings. They
must never be silently equated.

Passkey Stage B contains KIVI g64-r64 and CAGE r64, so that comparison receives
an exact method-level passkey join. It does not contain CAGE r128 or KIVI
g32-r128; that comparison is explicitly marked unavailable rather than filled
with another residual or group size.

## Metric directions and boundaries

- lower packed `paper_estimate.total_bytes` is better;
- lower local `joint_post_o_proj_mse` is better;
- negative paired delta NLL (CAGE minus KIVI) favors CAGE;
- higher passkey exact accuracy is better, with containment secondary.

The joined ledger retains the paired-PPL FP16 calibration failure and recorded
protocol deviation. Bootstrap intervals remain descriptive over deterministic
anchors. Memory is a packed paper estimate, not runtime tensor allocation,
CUDA peak, latency, throughput, or fused-kernel realized memory.

## Outputs

The synthesis produces eight operating-point rows, two comparison summaries,
an evidence protocol, a Markdown claim ledger, and an exact-length
memory-versus-PPL plot. The ledger is used to freeze the smallest mechanism
ablation matrix that can explain the observed operating-point behavior.
