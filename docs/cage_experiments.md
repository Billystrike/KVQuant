# CAGE Pareto pilot

This workflow targets the AutoDL Llama-2-7B stack. The model must be available at `/root/autodl-tmp/models/Llama-2-7b-hf`. TinyLlama is permitted only as a GPU debugging fixture for schema emission, layer counts, and cache reconstruction; it is not part of the experiment matrix and its output must not appear in paper-facing analysis.

## Prepare the prompt JSONL

Create `/root/autodl-tmp/cage_pilot_prompts.jsonl` with exactly one JSON object per line:

```json
{"sample_id":"doc-001","text":"Natural long-form text..."}
{"sample_id":"doc-002","text":"Different natural long-form text..."}
{"sample_id":"doc-003","text":"A third natural long-form text..."}
```

Each record must contain exactly `sample_id` and `text`. Both are non-empty strings, sample IDs are unique, and each text must tokenize with the configured Llama-2 tokenizer to at least 4096 tokens (the largest prompt length plus the teacher-forced continuation). The configured prompt lengths are 512, 1024, 2048, and 4095. The largest point uses 4095 prompt tokens plus one query token within Llama-2's native 4096-position context; no RoPE scaling or context extension is used. The runner adds special tokens, takes the first `T` tokens as the prompt, and token `T+1` as the continuation. It stores the sample ID and text SHA-256, not the raw text. Dataset acquisition is intentionally outside this runner.

## Run the matrices

From the repository root on the server, run the full 120-point pilot (3 samples x 4 lengths x 10 configurations):

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b.json --retry-transient-once
```

The output directory is `/root/autodl-tmp/cage_pareto_pilot`. Before the full pilot, run the checked-in six-point acceptance matrix (one sample x two lengths x FP16, KIVI `(32,32)`, and CAGE `R=32`):

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b_acceptance.json --retry-transient-once
```

The acceptance run requires a real AutoDL CUDA environment and Llama-2-7B. It passes only when all six run records are completed with no failure records; each run has 32 layer records; all mathematically non-negative metrics are finite and non-negative; FP16 joint errors are zero within numerical tolerance; Key and Value cache lengths equal `T`; the Key residual length is `T % R` and the Value residual length is `min(T,R)`; model-wide bytes equal the sum of all 32 layers; and an identical second invocation skips all six completed points. This GPU acceptance is required server-side and is not satisfied by the local CPU suite.

## Outputs and resume

The runner writes:

```text
/root/autodl-tmp/cage_pareto_pilot/
  manifest.resolved.json
  runs/<run_id>.json
  layers/<run_id>.jsonl
  failures/<run_id>.json
  summary/runs.jsonl
  summary/runs.csv
```

Each authoritative run JSON contains `schema_version`, `run_id`, `status`, `model`, `method`, `input`, `quantization`, `measurement`, `memory`, `metrics_aggregate`, `runtime_diagnostics`, and `provenance`. Each layer JSONL row identifies the run and layer and records method/prompt length, memory, reconstruction, Key, Value, and joint functional metrics. `summary/runs.jsonl` is authoritative; CSV is a flattened plotting convenience. `manifest.resolved.json` captures effective defaults and expanded jobs.

Run IDs include resolved configuration, model, source sample/hash, prompt length, and Git source state. Completed run files are atomic and are skipped on an identical rerun; completed points survive later failures. Only transient process failures are retried once with the flag above. OOM, input/configuration, and model-load failures are not silently treated as completed results. Changing code, input text, model, or configuration changes identity rather than reusing an incompatible result.

## Interpretation boundaries

Paper-facing memory is only `memory.paper_estimate`, derived from packed payload and metadata accounting. `memory.runtime_tensors` and `memory.cuda_peak_diagnostic` are separate diagnostics. CAGE currently uses fake quantization and may retain FP16 runtime tensors, so its runtime allocation, CUDA peak, and timing must never be presented as compressed-kernel memory or latency results.

This pilot measures local cache reconstruction and a memory-functional-error plane. It does not establish maintained language-model quality, retrieval accuracy, end-task quality, latency improvement, throughput improvement, or realized runtime compression. Those claims require separate downstream evaluation and a real compressed kernel.
