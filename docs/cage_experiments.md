# CAGE Pareto pilot

This workflow targets the AutoDL Llama-2-7B stack. The model must be available at `/root/autodl-tmp/models/Llama-2-7b-hf`. TinyLlama is permitted only as a GPU debugging fixture for schema emission, layer counts, and cache reconstruction; it is not part of the experiment matrix and its output must not appear in paper-facing analysis.

## Prepare the prompt JSONL

Create `/root/autodl-tmp/cage_pilot_prompts.jsonl` with exactly one JSON object per line:

```json
{"sample_id":"doc-001","text":"Natural long-form text..."}
{"sample_id":"doc-002","text":"Different natural long-form text..."}
{"sample_id":"doc-003","text":"A third natural long-form text..."}
```

Each record must contain exactly `sample_id` and `text`. Both are non-empty strings, sample IDs are unique, and each text must tokenize with the configured Llama-2 tokenizer to at least 4096 tokens (the largest prompt length plus the teacher-forced continuation). Manifest prompt lengths must also be unique, and two method IDs may not resolve to the same scientific method configuration. The configured prompt lengths are 512, 1024, 2048, and 4095. The largest point uses 4095 prompt tokens plus one query token within Llama-2's native 4096-position context. The worker loads model configuration metadata before tokenizer or weight construction, requires its actual `max_position_embeddings` to equal the resolved manifest value, and rejects non-null `rope_scaling`; no RoPE scaling or context extension is used. The runner adds special tokens, takes the first `T` tokens as the prompt, and token `T+1` as the continuation. It stores the sample ID and text SHA-256, not the raw text. Dataset acquisition is intentionally outside this runner.

## Run the matrices

From the repository root on the server, run the full 120-point pilot (3 samples x 4 lengths x 10 configurations):

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b.json --retry-transient-once
```

The output directory is `/root/autodl-tmp/cage_pareto_pilot`. Before the full pilot, run the checked-in six-point acceptance matrix (one sample x two lengths x FP16, KIVI `(32,32)`, and CAGE `R=32`):

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b_acceptance.json --retry-transient-once
```

The acceptance run requires a real AutoDL CUDA environment and Llama-2-7B. It passes only when all six run records are completed with no failure records; each run has 32 layer records; all mathematically non-negative metrics are finite and non-negative; FP16 error/divergence metrics are zero and FP16 top-k attention overlap is exactly `1.0`; Key and Value cache lengths equal `T`; the Key residual length is `T % R` and the Value residual length is `min(T,R)`; model-wide bytes equal the sum of all 32 layers; and an identical second invocation skips all six completed points. This GPU acceptance is required server-side and is not satisfied by the local CPU suite.

Bitwise GPU determinism is not claimed because strict deterministic algorithms are not enabled for this pilot: compatibility with the custom KIVI/FlashAttention CUDA path has not been established. On AutoDL, repeat the same six-point acceptance run from a source-identical checkout into a separate output directory (for example by copying the acceptance manifest outside the repository and changing only `output_dir`). Require identical schema, point identities, layer indices, and memory integers, then compare every run-level and layer-level metric using `abs(repeat - reference) <= 1e-7 + 1e-5 * abs(reference)`. Record and investigate any violation before running the full matrix.

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

Each authoritative run JSON uses schema version 1 and exactly these top-level fields: `schema_version`, `run_id`, `status`, `model`, `method`, `input`, `quantization`, `measurement`, `memory`, `metrics_aggregate`, `runtime_diagnostics`, and `provenance`. Each schema-version-1 layer row has the matching run/layer/method/prompt/phase/query context, every required finite non-negative metric, and `memory.cache_structure.key` / `.value` fields for total, quantized-history, and FP16-residual tokens. Unknown run top-level or layer fields are rejected. `summary/runs.jsonl` is authoritative; CSV is a flattened plotting convenience. `manifest.resolved.json` captures effective defaults and expanded jobs.

`run_id` contains only point-local scientific identity. It excludes job/matrix IDs, sample/length selection lists, prompt-file paths, and output paths, so the six acceptance points are the same points in the full matrix and are reused rather than duplicated. A point is resumable only when both its run JSON and matching non-empty layer JSONL pass the shared strict validator, including contiguous layer indices, context, metrics, cache partitions, internal cache-byte arithmetic, and model-wide byte sums. Aggregate metric and statistic names must exactly match schema 1. Mean, median, and maximum are recomputed from layer rows and compared with `rel_tol=1e-12` and `abs_tol=1e-12`. For both `paper_estimate` and `runtime_tensors`, `payload_only_bytes` is Key plus Value payload, `metadata_bytes` is the four scale/min-or-zero-point fields, and `total_bytes` is all payload, scale/min-or-zero-point, bucket-index, and residual-FP16 fields. CUDA peaks are never cache-byte inputs. Invalid completed artifacts are rerun and atomically replaced. Aggregation validates and summarizes only completed run IDs expected by the current resolved manifest, selected prompt texts, and source state; valid historical artifacts remain on disk but are excluded, while an invalid expected artifact rejects the summary update.

Before tokenizer or model construction, the worker strictly revalidates `manifest.resolved.json`, including exact resolved method configuration, unique prompt lengths, no duplicate resolved scientific method configurations, the scoped CAGE constraints, and any embedded expanded jobs. It then validates actual native model configuration before tokenizer and weight loading. Invalid manifests or native-context preflight exit with code 2.

Worker exit codes are 0 success, 2 shared deterministic manifest/input/preflight error, 3 CUDA OOM, 4 model construction/load error, 5 positively identified transient unusable process/model state, and 6 isolated deterministic point/job error after shared preflight/model load. Codes 2 and 6 are non-retryable. A usable worker continues later points after code 6, and the matrix records a nonzero result but continues later jobs; code 2 stops later jobs. Only code 5 is marked retryable and retried once by `--retry-transient-once`.

Run IDs include resolved configuration, model, source sample/hash, prompt length, and Git source state. The conservative source identity includes all tracked dirty state; any tracked repository change intentionally invalidates reuse. There is no experiment-path allowlist. Completed run files are atomic and are skipped on an identical rerun; completed points survive later failures. Only transient process failures are retried once with the flag above. OOM, input/configuration, model-load, and isolated deterministic failures are not silently treated as completed results. Changing code, input text, model, or configuration changes identity rather than reusing an incompatible result.

## Interpretation boundaries

Paper-facing memory is only `memory.paper_estimate`, derived from the prefill-T packed payload and metadata accounting. `memory.runtime_tensors` is also a prefill-T cache snapshot. `memory.cuda_peak_diagnostic` is a separate whole-point diagnostic updated after decode and metric reconstruction; it does not summarize the T+1 cache. CAGE currently uses fake quantization and may retain FP16 runtime tensors, so its runtime allocation, CUDA peak, and timing must never be presented as compressed-kernel memory or latency results.

The scoped pilot accepts only CAGE `k_bits == v_bits == 2`, both Key and Value enabled, Key importance `q2_var`, and Value importance `wo_var`. Key-only, Value-only, 4-bit, and alternate-policy experiments are later ablations and are rejected here. NaN or infinity in source tensors, cache reconstruction, logits, outputs, serialized metrics, or aggregates fails validation; it is never converted to a plausible zero.

Provenance captures the resolved seed, whether deterministic algorithms are enabled, deterministic warn-only state when supported, cuDNN deterministic and benchmark flags, and `CUBLAS_WORKSPACE_CONFIG`. Production code records these settings but does not call `torch.use_deterministic_algorithms(True)`.

This pilot measures local cache reconstruction and a memory-functional-error plane. It does not establish maintained language-model quality, retrieval accuracy, end-task quality, latency improvement, throughput improvement, or realized runtime compression. Those claims require separate downstream evaluation and a real compressed kernel.
