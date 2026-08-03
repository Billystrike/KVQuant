# Qwen3-8B CAGE/Kitty matched-memory protocol

## Purpose and freeze point

This protocol was frozen after Kitty environment, runtime, accuracy-simulation,
packed-memory, and repeat-resume acceptance, but before implementing CAGE for
Qwen3 or observing any formal Qwen3 CAGE quality result. Its purpose is to
maximize the power of scientifically motivated tests of CAGE without selecting
budgets, samples, or endpoints after seeing outcomes.

The machine-readable authority is
`configs/qwen3_8b_matched_memory_protocol_v1.json`. If this document and the
JSON disagree, the JSON is authoritative.

## Two matrices, not one nominal 2-bit comparison

The official reproduction matrix contains FP16, KIVI-2, KIVI*-2, Kitty, and
Kitty-Pro under the Kitty paper definitions. Kitty promotes 12.5% (16/128) of
Key channels to INT4; Kitty-Pro promotes 25% (32/128). KIVI-2 uses group and
buffer size 128 without a sink, while KIVI*-2 additionally preserves 32 sink
tokens. This matrix checks reproduction and must not be described as
memory-matched.

The matched-memory matrix is separate. It uses exact prompt lengths 1024,
2048, and 4032 and admits a pair only when complete active packed bytes differ
from the Kitty or Kitty-Pro target by at most 3%. The 512-token point is absent
because the natural 32-token candidate grid cannot match either target within
that tolerance. No later quality result may be used to add it or relax the
gate.

## Complete packed-byte scope

The primary budget is active packed cache storage at the exact prompt length.
It includes packed K/V payload, FP16 scale and min/zero-point metadata,
persistent channel or bucket indices, occupied FP16 residual/sink/Q/local
storage, and used page-table entries. It excludes model weights, temporary
quantization workspace, fake-quant FP16 storage, CUDA allocator rounding,
whole-run CUDA peaks, and unused preallocated capacity.

CAGE retains its conservative current representation assumption: separate Key
and Value `torch.long` bucket permutations. For Qwen3-8B this is 589,824 bytes
model-wide. It must not be silently replaced by an ideal uint8 layout before a
real packed CAGE implementation exists.

Kitty active payload, runtime-reserved logical storage, and observed CUDA
allocation remain separate quantities. The frozen Kitty packed-memory audit is
SHA-256
`91275559aabceea6d460ccbfa8b478da4b6273f53af6d8cdeba4cc804ba477a2`.

The independent formula implementation is derived from the frozen Kitty
runtime source snapshot whose SHA-256 is
`4d2b919c9f9e455a2bfd34f896dfda8cd6955c2f4e00885a43d855881494d8db`.
It accounts separately for the low Key bits, promoted high Key bits, per-page
channel indices, Key and Value scale/zero-point tensors, Value payload, used
page-table entries, and occupied sink, Key Q-buffer, Value Q-buffer, and Value
local-window tokens. The validator must reproduce the frozen Kitty audit hash,
all six target byte totals, and every selected CAGE/KIVI point before formal
quality runs begin.

## Deterministic matching rule

CAGE candidates use residual lengths 32 through 512 in increments of 32 and
the frozen adaptive group sizes `[32, 64, 128]`. KIVI candidates use group
sizes 32, 64, and 128, with residual lengths that are multiples of the chosen
group size through 512.

For each length and target, selection minimizes absolute relative packed-byte
difference without consuming a quality result. Exact ties choose the smaller
byte total, then the smaller residual, then the smaller group. A selected pair
is valid only at an absolute relative difference of at most 3%. Consequently,
there is no KIVI pair for the 2048-token Kitty (12.5%) target; the closest
candidate misses the gate, and the comparison remains absent.

The exact selected method IDs, bytes, and signed differences are frozen in the
JSON rather than repeated manually here.

## Independent validation result

The complete CPU-only recalculation passed on commit
`ecf265f9c5dec36e775b024e2331ce0b982cebd7`. Six unit tests reproduced the
Kitty and Kitty-Pro page layouts, the 320-token probe, all six target totals,
the selected CAGE/KIVI totals, and the deterministic byte-only selection rule.
The validator matched the frozen Kitty audit SHA-256, confirmed the
589,824-byte model-wide CAGE index charge, and correctly rejected the closest
KIVI candidate for the 2048-token Kitty target at a 3.4322% relative excess.
All checks passed with no failures. The validation JSON SHA-256 is
`6301966720f690fe136c4dbe9e83546ac1923832811690504cdc9e2563453bee`
and the complete log SHA-256 is
`89fdb838c895624110c19179909afca6405ac3047ee02b305ad42b3f9ac4a3cf`.

## CAGE-centered hypotheses

The primary scientific claim under test is not that every CAGE configuration
beats every nominally 2-bit method. It is that adaptive group-size allocation
can preserve cache-conditioned quality more effectively than competing cache
allocations at the same complete packed-memory budget.

Primary CAGE-minus-baseline endpoints are paired continuation mean NLL, Key
and Value reconstruction error, and attention-output perturbation. Negative
quality/error deltas favor CAGE. The design intentionally emphasizes these
mechanism-aligned endpoints, where the Llama-2 evidence predicts CAGE should
have power, while keeping all cases paired on raw input, token IDs, length,
continuation targets, anchor identity, and scoring.

Full CAGE must also be compared with same-budget fixed-random and uniform
assignment controls plus the Key-only and Value-only adaptive mechanism
variants. This distinguishes an adaptive-allocation benefit from merely
spending a different number of bytes.

Passkey remains a saturation diagnostic and cannot be promoted to primary
advantage evidence. Unfavorable primary results remain in the package. Sample,
budget, endpoint, or tolerance changes after observing Qwen3 results require a
dated deviation record and cannot replace the frozen primary analysis.

## Claim boundary

Kitty accuracy simulation and CAGE remain fake-quant quality paths. Only
Kitty-Pro currently has official top-level packed Triton runtime acceptance.
This protocol therefore supports packed paper-estimate and quality claims, not
CAGE latency, CAGE throughput, realized CAGE CUDA-memory savings, or official
Kitty-12.5 runtime-performance claims.
