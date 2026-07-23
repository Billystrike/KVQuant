# CAGE-KV Llama-2-7B Pareto Pilot Design

## 1. Purpose

This design defines the first reproducible paper-oriented experiment for CAGE-KV on Llama-2-7B. The pilot measures the relationship between paper-facing KV-cache memory and local attention functional error for FP16, KIVI, and default CAGE-KV.

The pilot is diagnostic rather than a final quality evaluation. Its results may support method development and the design of later experiments, but paper-level quality claims must additionally use downstream metrics such as perplexity, passkey retrieval, QA accuracy, and LongBench.

## 2. Scientific objective

The primary comparison is a memory-quality Pareto analysis. It does not compare only the default configurations of each method.

The two axes are:

- paper-facing packed KV-cache bytes, with payload, metadata, bucket indices, and residual FP16 storage reported separately;
- local functional error relative to FP16 attention, measured at a teacher-forced decode step.

The main functional-error metric is joint post-output-projection error. Key, Value, and reconstruction metrics remain explanatory metrics rather than being collapsed into an undocumented composite score.

## 3. Scope

### 3.1 Included

- Llama-2-7B only.
- FP16, original KIVI, and default CAGE-KV.
- External JSONL natural-text inputs.
- Prompt lengths of 512, 1024, 2048, and 4095 tokens.
- KIVI group-size and residual sweeps.
- CAGE residual sweep using the default bucket group profile.
- Paper-facing cache accounting and separate runtime tensor diagnostics.
- Per-layer and aggregated perturbation metrics against a common FP16 reference.
- Deterministic execution, provenance capture, failure records, and resumable matrix execution.

### 3.2 Excluded

- Key-only, Value-only, no-clipping, uniform-bucket, and other ablations.
- Passkey retrieval, perplexity, QA, and LongBench.
- Latency or throughput claims.
- A fused or packed CAGE CUDA kernel.
- Mistral, Qwen, or other model-family integration.
- Treating fake-quant runtime GPU memory as the memory footprint of a final compressed implementation.

TinyLlama may be used as an optional GPU debugging fixture. It is not part of the experiment matrix, paper tables, or detailed result analysis.

## 4. Required semantic alignment

### 4.1 Prefill behavior

KIVI computes prefill attention using FP16 Key and Value tensors and quantizes only the tensors written to the cache. CAGE must use the same semantic boundary:

- prefill attention uses FP16 Key and Value tensors;
- CAGE fake quantization affects the stored cache;
- quantization affects later decode attention through the reconstructed cache.

The current CAGE behavior, which uses fake-quantized tensors in prefill attention, must not be used for the Pareto pilot.

### 4.2 Residual buffer behavior

CAGE must match the operational buffer behavior of the repository's KIVI implementation:

- Key history is flushed in blocks controlled by `residual_length`; after prefill, its FP16 remainder contains `prompt_length % residual_length` tokens;
- Value history retains the most recent `min(prompt_length, residual_length)` tokens in FP16;
- decode updates follow the same Key flush and Value rolling-buffer behavior.

The comparison uses the actual stored cache structure. It does not assume that both Key and Value always retain exactly `residual_length` FP16 tokens.

### 4.3 Common FP16 reference

All methods use the same metric definition and the same FP16 query source. For each input, an FP16 reference pass over T+1 tokens captures the final-position query at every layer. At the teacher-forced decode measurement, each method uses that captured query to compare attention computed with the FP16 history against attention computed with the method's reconstructed history.

- FP16 reconstructs the identical FP16 history and therefore has zero quantization error.
- KIVI reconstructs its actual packed Key and Value cache using the stored codes and metadata.
- CAGE reconstructs its bucketed fake-quant cache and FP16 residual buffers.

Metrics must not compare CAGE prefill perturbation against KIVI's unperturbed prefill path.

Candidate-path queries are not used for the primary pilot metric because upstream quantization errors would make the query tensor differ across methods. End-to-end propagation is deferred to downstream quality evaluations rather than mixed into this controlled local metric.

## 5. Architecture

The existing `scripts/cage_smoke.py` remains a short generation sanity check and is not expanded into a paper experiment runner.

### 5.1 Matrix orchestrator

`scripts/cage_run_matrix.py` will:

- read and validate a JSON manifest;
- resolve defaults and write `manifest.resolved.json`;
- expand method configurations;
- launch one isolated worker process per model/method/config job;
- skip completed experiment points using stable run identifiers;
- retry only eligible transient process failures;
- aggregate completed records into JSONL and CSV summaries.

### 5.2 Experiment worker

`scripts/cage_experiment_worker.py` will load a model once for one resolved method configuration, then process the assigned samples and prompt lengths sequentially. Each sample-length combination remains an independently identified and atomically written experiment point.

The worker will:

1. load and tokenize the requested natural-text records;
2. take the first T tokens as the prompt and token T+1 as the teacher-forced continuation;
3. run an FP16 reference pass over T+1 tokens with `use_cache=False` and capture each layer's final-position query;
4. run FP16 prefill attention over T tokens while constructing the method-specific cache;
5. run one teacher-forced candidate decode step;
6. collect per-layer perturbation metrics using the captured FP16 query, FP16 history, and reconstructed candidate history;
7. summarize each layer's actual cache structure;
8. record runtime tensor and CUDA peak diagnostics separately;
9. atomically write the run and layer records.

Grouping points by method configuration avoids loading Llama-2-7B once for every sample and prompt length. Process isolation is retained between configurations, where allocator state and model configuration differ.

### 5.3 Manifest

`configs/cage_pilot_llama2_7b.json` will contain:

- model reference and dtype;
- input JSONL path and sample selection;
- prompt lengths;
- method configurations;
- metric protocol;
- deterministic seed;
- output directory;
- retry and resume policy.

The resolved manifest, rather than the shell command alone, is the complete experiment specification.

### 5.4 Metric and cache helpers

The metric implementation will support a neutral FP16-reference interface rather than assuming that every candidate is CAGE. It will accept the captured FP16 decode query, FP16 Key/Value history, reconstructed Key/Value history, output-projection weights, attention mask, importance tensors where applicable, and method metadata.

KIVI cache reconstruction must use its actual packed cache and metadata. CAGE reconstruction must combine bucket payloads with its Key and Value FP16 buffers. Memory summaries must be derived from the same cache records used by attention and metrics.

## 6. Pilot experiment matrix

The default pilot contains three distinct natural-text samples.

| Dimension | Values |
|---|---|
| Model | Llama-2-7B |
| Samples | 3 natural-text documents |
| Prompt lengths | 512, 1024, 2048, 4095 |
| FP16 | one configuration |
| KIVI `(group_size, residual_length)` | `(32,32)`, `(32,64)`, `(32,128)`, `(64,64)`, `(64,128)`, `(128,128)` |
| CAGE bucket groups | `[32, 64, 128]` |
| CAGE residual length | 32, 64, 128 |
| Key/Value bits | 2 |
| Decode protocol | one teacher-forced continuation token |
| Randomness | deterministic, no repeated pilot seeds |

Only KIVI pairs satisfying `residual_length % group_size == 0` are included because the decode-time Key flush requires this invariant. This produces 120 experiment points:

`3 samples * 4 lengths * (1 FP16 + 6 KIVI + 3 CAGE)`.

The first server validation may restrict the same manifest to one sample, prompt lengths 512 and 2048, and these configurations:

- FP16;
- KIVI with group size 32 and residual length 32;
- CAGE default with residual length 32.

This reduced validation is not a separate scientific matrix.

## 7. Metric protocol

### 7.1 Measurement phase

Metrics are collected for one teacher-forced decode position after a prefill of exactly T tokens. A separate FP16 T+1-token pass supplies a common final-position query for every method. Metric computation therefore avoids constructing a full `T x T` FP32 diagnostic attention matrix while measuring cache error at a decode position.

The natural text must contain at least T+1 tokens. The continuation token is taken from the source text and is not sampled from the model. The output records `query_source=fp16_reference_final_position` so the control variable is explicit.

For Llama-2's native 4096-position context, the largest point is a 4095-token prompt plus one teacher-forced query token. Manifest validation requires `prompt_length + measurement.decode_tokens <= model.max_position_embeddings`; no RoPE scaling or context extension is used.

### 7.2 Existing explanatory metrics

The pilot retains:

- relative Key reconstruction error;
- attention-logit MSE;
- attention-score KL divergence;
- top-k attention overlap;
- weighted Key error;
- relative Value reconstruction error;
- Value-only attention-output MSE;
- Value-only post-output-projection MSE;
- weighted Value error.

### 7.3 Joint functional metrics

The pilot adds metrics that apply the perturbed attention scores to the reconstructed Value cache:

- `joint_attention_output_mse`;
- `joint_post_o_proj_mse`;
- `joint_attention_output_relative_error`.

`joint_post_o_proj_mse` is the primary local functional-error axis. Results must also report the joint relative error and the Key/Value explanatory metrics. The pilot must not present this local metric as a replacement for downstream task quality.

### 7.4 Aggregation

Every layer produces an individual record. Run-level records contain exactly the schema-1 metric names and, for each metric, exactly the mean, median, and maximum across layers. Completed-point validation recomputes these values with the worker's canonical definitions and compares them using `rel_tol=1e-12` and `abs_tol=1e-12`.

## 8. Memory protocol

### 8.1 Paper-facing estimate

`memory.paper_estimate` reports the packed representation that a completed implementation would use:

- Key payload bytes;
- Value payload bytes;
- Key scale bytes;
- Value scale bytes;
- Key minimum or zero-point bytes;
- Value minimum or zero-point bytes;
- bucket-index bytes;
- residual FP16 bytes;
- payload-only bytes;
- metadata bytes;
- total cache bytes.

Layer values and the model-wide sum across all 32 layers are both retained.

### 8.2 Runtime diagnostics

`memory.runtime_tensors` reports the bytes occupied by tensors actually stored by the prototype. `memory.cuda_peak_diagnostic` reports peak allocated and reserved CUDA memory for the whole experiment point, sampled after decode and metric reconstruction. Paper/runtime cache byte summaries remain snapshots of the prefill cache at exactly T tokens; the T+1 decode cache is never used for those summaries.

These namespaces must remain separate. The fake CAGE tensors may be FP16 and must not be presented as a compressed runtime implementation. Runtime timing and peak memory are diagnostics, not evidence of CAGE latency or implementation-level memory advantages.

## 9. Input contract

The external JSONL format is:

```json
{"sample_id": "doc-001", "text": "Natural long-form text..."}
```

Requirements:

- `sample_id` is non-empty and unique;
- `text` is non-empty;
- tokenized text contains at least the maximum requested prompt length plus one token;
- input is truncated deterministically after tokenization;
- the output stores the sample identifier and SHA-256 of the text, not a duplicate of the raw text.

No dataset download is built into the pilot runner.

## 10. Output contract

The output layout is:

```text
output_dir/
  manifest.resolved.json
  runs/<run_id>.json
  layers/<run_id>.jsonl
  failures/<run_id>.json
  summary/runs.jsonl
  summary/runs.csv
```

### 10.1 Run record

Each authoritative run record contains:

- `schema_version`;
- `run_id`;
- `status`;
- `model`;
- `method`;
- `input`;
- `quantization`;
- `measurement`;
- `memory`;
- `metrics_aggregate`;
- `runtime_diagnostics`;
- `provenance`.

The resolved method configuration records all effective values, including defaults. Provenance includes the Git commit, dirty-worktree flag, Python, PyTorch, Transformers, CUDA, driver and GPU versions, resolved deterministic seed, deterministic-algorithm and warn-only state, cuDNN deterministic and benchmark flags, `CUBLAS_WORKSPACE_CONFIG`, and a normalized command summary. These values are recorded; strict deterministic algorithms are not enabled by production code and bitwise GPU determinism is not claimed.

### 10.2 Layer record

Each layer JSONL record contains:

- integer `schema_version == 1`, run identifier, and layer index;
- method and prompt length;
- layer memory summary;
- `memory.cache_structure.key` and `.value`, each with `total_tokens`, `quantized_history_tokens`, and `fp16_residual_tokens`;
- reconstruction metrics;
- Key functional metrics;
- Value functional metrics;
- joint functional metrics.

Version 1 uses exact run top-level and layer-row field sets; unknown fields are rejected. Extension requires a new documented schema version rather than ad hoc fields.

### 10.3 Stable identifiers and derived summaries

The stable run identifier is derived only from point-local scientific identity: normalized model identity, method name and fully resolved method configuration, measurement protocol including seed, sample identifier and text hash, prompt length, and source-state identity. Matrix/storage fields (`job_id`, sample/length selection lists, prompt-file path, output directory, and matrix IDs) are excluded. Therefore overlapping acceptance/full points share identifiers and acceptance output is reused in the common output directory. Source-state identity contains the Git commit and, when the worktree is dirty, a deterministic hash of all tracked dirty state plus included untracked experiment-code changes. This deliberately invalidates reuse after any tracked repository change; no experiment-path allowlist is used. Completed run records are written atomically.

`summary/runs.jsonl` is the canonical combined summary. CSV is a flattened plotting convenience and is not the authoritative source for nested experiment data.

## 11. Validation and failure handling

### 11.1 Preflight validation

Before model loading, the runner validates:

- JSONL schema and unique sample identifiers;
- unique prompt lengths and unique fully resolved scientific method configurations;
- sufficient token length without silent shortening;
- actual native model context equal to the resolved manifest limit, including the prompt and decode/query token, with non-null RoPE scaling rejected before tokenizer and weight loading;
- method name and known manifest fields;
- bit width, group sizes, residual length, bucket count, and method-specific constraints; scoped CAGE requires 2-bit Key and Value, both sides enabled, `q2_var` Key importance, and `wo_var` Value importance;
- completeness of the resolved configuration;
- the KIVI decode invariant `residual_length % group_size == 0`.

Unknown fields and invalid configurations fail explicitly.

### 11.2 Failure isolation and resume

- A model/method/config job runs in its own process.
- Completed experiment points survive a later failure in the same job.
- Resume skips only a version-1 run JSON whose matching non-empty layer JSONL passes the complete row-count, index, context, metric, canonical aggregate, cache-structure, internal cache-byte arithmetic, and recursive memory-sum validator. Payload-only, metadata, bucket-index, residual-FP16, and total byte relationships are checked for layer and model `paper_estimate` and `runtime_tensors`; CUDA peaks are not cache bytes. Invalid artifacts are rerun and atomically overwritten. Aggregation uses the same validator and validates all completed points before replacing either summary.
- Exit 2 is a shared deterministic manifest/input/preflight error, 3 is CUDA OOM, 4 is model construction/load failure, 5 is a positively identified transient unusable process/model state, and 6 is an isolated deterministic point/job failure after shared preflight/model load. Failure-record `retryable` is true exactly for code 5. A usable worker and the matrix continue after code 6; code 2 stops later matrix jobs.
- Only code 5 is retried automatically, at most once.
- A deterministic OOM for an unchanged configuration is not retried in a loop.
- Non-completed run records are ignored by aggregation; invalid completed records are rejected.
- Non-finite source, cache, reconstruction, logit, output, or final metric values fail the point explicitly and never produce a completed record.

## 12. Testing strategy

### 12.1 CPU tests

CPU unit tests cover:

- manifest validation and matrix expansion;
- stable run identifiers and resume behavior;
- JSONL parsing and exact token slicing;
- continuation-token selection;
- FP16 final-position query capture and reuse across methods;
- atomic result writing and aggregation;
- joint functional metric numerics and tensor-shape validation;
- memory-field totals and separation of paper and runtime namespaces;
- CAGE Key buffer length equal to `T % R`;
- CAGE Value buffer length equal to `min(T, R)`;
- CAGE prefill attention remaining independent of cache fake quantization.

### 12.2 Optional debug model

TinyLlama may be used to shorten GPU debugging of schema emission, layer-record counts, and cache reconstruction. Passing this debug check is not an experimental result and is not required in paper-facing reports.

### 12.3 Llama-2-7B acceptance run

The required GPU acceptance run uses:

- one natural-text sample;
- prompt lengths 512 and 2048;
- FP16;
- KIVI with group size 32 and residual length 32;
- CAGE default with residual length 32.

Acceptance criteria:

- every metric is finite and non-negative where mathematically required;
- FP16 quantization functional error is zero within the defined numerical tolerance;
- KIVI and CAGE cache lengths equal the prompt length;
- Key and Value residual buffers follow the approved operational rules;
- every Llama-2-7B layer produces one layer record;
- paper-facing and runtime fake memory remain separate;
- model-wide memory equals the sum of layer records;
- rerunning the same resolved manifest skips completed points.
- repeating the six points from a source-identical checkout into a separate output directory preserves structural/memory values exactly and satisfies `abs(repeat - reference) <= 1e-7 + 1e-5 * abs(reference)` for every run/layer metric; this is a server acceptance step, not a bitwise-determinism claim.

## 13. Interpretation boundaries

The pilot can establish that the experiment infrastructure is reproducible and that CAGE occupies promising locations in a local memory-functional-error plane. It cannot by itself establish maintained language-model quality, retrieval ability, latency improvement, or realized runtime compression.

The next experiment stages, each designed separately, are:

1. CAGE ablations;
2. perplexity;
3. passkey retrieval;
4. selected LongBench tasks;
5. final memory-quality Pareto analysis using downstream quality metrics.
