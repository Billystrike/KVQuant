# Qwen3-8B formal results receipt and analysis v1

## Post-execution freeze

Both full formal partitions completed on source commit
`f0fbcb7efc19f6b87d6afa6ac5e79749b30204db` under the previously frozen
protocol, execution config, input manifest, and repeated-acceptance gate. This
record fixes their artifact identities before any joint quality
interpretation. The machine-readable authority is
`configs/qwen3_8b_formal_results_receipt_v1.json`.

The CAGE partition contains 1,000 fresh cases and no failures: 150 FP16, 650
CAGE, and 200 KIVI cases over 20 method-length points and 50 anchors. Its
scientific-payload SHA-256 is
`612a3757e49e6b4a7bd5eff96eeff7087934b0bdac1455b7c91cf8cd97cc2332`.
Its complete post-run audit JSON SHA-256 is
`de96d5d7dffbc6d4a108e11b4c41415e770f186cba124b31e96075f43d2cc41f`.

The Kitty partition contains 300 fresh cases and no failures: Kitty and
Kitty-Pro at three prompt lengths, 50 anchors per point. Its scientific-payload
SHA-256 is
`5784d37b15934e9e95923460157cc17dee00f5aacd0017dd4177ab372d24f157`.
Its complete post-run audit JSON SHA-256 is
`9534d3c3d2f72820f4bdc7eb010f0048e999fce994ed86b0589c2a0e8f7f3b65`.

Together they contain the protocol's exact 1,300 unique cases and 83,200
continuation-token NLL targets. No failed or resumed case contributes to this
result set.

## Analysis freeze

The analysis implementation was fixed before loading the joint quality
values. Each comparison is paired by the 50 deterministic anchor indices and
uses candidate mean NLL minus baseline mean NLL; a negative delta favors the
candidate. Exact zero is the only tie.

Every contrast independently reinitializes Python `random.Random` with seed
20260803, draws 10,000 bootstrap samples of 50 paired anchor deltas with
replacement, and reports the Type-7 linearly interpolated 2.5th and 97.5th
percentiles. These intervals describe stability over the deterministic corpus
grid and are not population-significance intervals.

The ten pre-registered matched-memory comparisons are emitted in protocol
order. At each of the two mechanism points, the analysis emits full CAGE
against all four exact-byte controls, plus Key-only and Value-only adaptation
against fixed-uniform. The base Pareto calculation minimizes complete active
packed bytes and token-weighted mean NLL and does not include mechanism
variants as additional budget candidates.

## Execution

After pulling the analysis commit on the server, run the analyzer in the
isolated CAGE/Qwen environment. It is CPU/file-bound and does not load model
weights:

```bash
cd /root/autodl-tmp/KVQuant
/root/miniconda3/bin/conda run --no-capture-output \
  -p /root/autodl-tmp/conda-envs/cage-qwen3 \
  python scripts/qwen3_analyze_formal_results.py \
  --receipt configs/qwen3_8b_formal_results_receipt_v1.json \
  --output-dir /root/autodl-tmp/qwen3_formal/analysis_v1
```

The output directory contains the full aggregate JSON, method-length CSV,
primary-comparison CSV, mechanism-comparison CSV, paper-table Markdown, and a
manifest that hashes every derived output. The analyzer fails closed on an
artifact, identity, case-set, score-recalculation, scientific-payload, or
memory-formula mismatch.

## Claim boundary

The derived perplexities are exponentiated cache-conditioned continuation
mean NLL values, not canonical full-corpus WikiText-2 perplexity. Memory is the
complete active packed paper estimate fixed by the matched-memory protocol.
The results do not support CAGE fused-kernel latency, throughput, or realized
CUDA-memory claims. Passkey remains a saturation diagnostic.
