# CAGE-KV Pareto Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Llama-2-7B pilot that compares FP16, KIVI, and default CAGE-KV on paper-facing cache memory versus controlled local functional error using a shared FP16 decode query.

**Architecture:** Keep `cage_smoke.py` unchanged. Correct CAGE cache semantics in the Llama path, add neutral reconstruction/metric/memory helpers, then build a manifest-driven worker and a subprocess orchestrator. Each method configuration loads once, each sample-length point writes atomically, and JSON is canonical while CSV is derived.

**Tech Stack:** Python 3.10, PyTorch 2.4.1+cu121, Transformers 4.43.1, existing KIVI Triton/CUDA extension, `unittest`, JSON/JSONL/CSV from the standard library.

## Global Constraints

- Experiment model: `/root/autodl-tmp/models/Llama-2-7b-hf`; TinyLlama is optional debug-only and never paper-facing.
- Prompt lengths: 512, 1024, 2048, 4096; each natural-text record must contain at least T+1 tokens.
- Methods: FP16; six valid KIVI `(group_size, residual_length)` pairs; CAGE default bucket groups `[32,64,128]` with residual lengths 32, 64, 128.
- CAGE prefill attention must use FP16 K/V; only stored history is fake-quantized.
- CAGE Key buffer length is `T % R`; Value buffer length is `min(T, R)`.
- Primary metric query is the final-position query captured from an FP16 T+1-token reference pass.
- `memory.paper_estimate`, `memory.runtime_tensors`, and `memory.cuda_peak_diagnostic` must never be merged.
- Do not claim latency, throughput, realized compressed runtime memory, or downstream task quality.
- Preserve `scripts/cage_smoke.py` behavior and all existing tests.

---

### Task 1: Align CAGE Prefill and Residual Buffers with KIVI

**Files:**
- Modify: `models/llama_kivi.py:84` (`_cage_fake_forward`)
- Modify: `tests/test_cage_attention_integration.py`

**Interfaces:**
- Consumes: existing `CageKeyCache`, `CageValueCache`, bucket assignment, and fake quantizers.
- Produces: CAGE caches whose quantized prefix plus FP16 suffix equals `kv_seq_len`; prefill output computed from unquantized K/V.

- [ ] **Step 1: Replace the prefill expectations with failing semantic tests**

Add tests that compare identical attention modules with quantization enabled/disabled and assert the approved buffers:

```python
def test_cage_prefill_attention_is_fp16_and_cache_uses_kivi_buffers(self):
    torch.manual_seed(0)
    quantized = self._attention()
    reference = self._attention()
    reference.load_state_dict(quantized.state_dict())
    reference.cage_config.cage_k_enable = False
    reference.cage_config.cage_v_enable = False
    hidden = torch.randn(1, 5, 16)
    positions = torch.arange(5).unsqueeze(0)
    mask = torch.zeros(1, 1, 5, 5)

    actual, _, cache = quantized(hidden, attention_mask=mask, position_ids=positions, use_cache=True)
    expected, _, _ = reference(hidden, attention_mask=mask, position_ids=positions, use_cache=False)

    torch.testing.assert_close(actual, expected)
    unpacked = unpack_cage_past_key_value(cache)
    self.assertEqual(unpacked.key_cache.key_quant_buckets[0].shape[-2], 4)
    self.assertEqual(unpacked.key_cache.key_full.shape[-2], 1)
    self.assertEqual(unpacked.value_cache.value_quant_buckets[0].shape[-2], 3)
    self.assertEqual(unpacked.value_cache.value_full.shape[-2], 2)

def test_cage_decode_flushes_key_block_and_rolls_value_buffer(self):
    attention = self._attention()
    prefill = torch.randn(1, 5, 16)
    _, _, cache = attention(
        prefill,
        attention_mask=torch.zeros(1, 1, 5, 5),
        position_ids=torch.arange(5).unsqueeze(0),
        use_cache=True,
    )
    _, _, updated = attention(
        torch.randn(1, 1, 16),
        attention_mask=torch.zeros(1, 1, 1, 6),
        position_ids=torch.tensor([[5]]),
        past_key_value=cache,
        use_cache=True,
    )
    unpacked = unpack_cage_past_key_value(updated)
    self.assertIsNone(unpacked.key_cache.key_full)
    self.assertEqual(unpacked.key_cache.key_quant_buckets[0].shape[-2], 6)
    self.assertEqual(unpacked.value_cache.value_quant_buckets[0].shape[-2], 4)
    self.assertEqual(unpacked.value_cache.value_full.shape[-2], 2)
```

- [ ] **Step 2: Run the tests and verify the old semantics fail**

Run: `python -m unittest tests.test_cage_attention_integration -v`

Expected: FAIL because prefill currently attends to fake-quantized tensors and retains no approved residual suffix.

- [ ] **Step 3: Implement prefix/suffix splitting and incremental flushes**

Add these helpers to `LlamaAttention_KIVI` and use them in `_cage_fake_forward`:

```python
def _split_cage_key_history(self, states):
    remainder = states.shape[-2] % self.residual_length
    if remainder == 0:
        return states, None
    return states[:, :, :-remainder, :].contiguous(), states[:, :, -remainder:, :].contiguous()

def _split_cage_value_history(self, states):
    keep = min(states.shape[-2], self.residual_length)
    if states.shape[-2] == keep:
        return None, states
    return states[:, :, :-keep, :].contiguous(), states[:, :, -keep:, :].contiguous()

def _append_cage_bucket_payloads(self, old_buckets, new_states, indices):
    if new_states is None or new_states.shape[-2] == 0:
        return old_buckets
    new_buckets = self._gather_bucket_payloads(new_states, indices)
    return tuple(
        new if old is None else torch.cat((old, new), dim=2)
        for old, new in zip(old_buckets, new_buckets)
    )
```

In the prefill branch, compute importance and assignments from FP16 tensors, split each side, fake-quantize only the quantized prefix, and pass the original `key_states`/`value_states` to `_cage_compute_attention`. In decode, reconstruct the previous prefix and suffix for attention, then append the current Key to `key_full`, flushing exactly R tokens when full; append the current Value to `value_full`, quantizing only the overflow and retaining the last R tokens.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_cage_attention_integration tests.test_cage_cache tests.test_cage_memory -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/llama_kivi.py tests/test_cage_attention_integration.py
git commit -m "fix: align CAGE cache buffers with KIVI"
```

---

### Task 2: Add Joint Metrics and Cache Reconstruction

**Files:**
- Modify: `utils/cage_metrics.py`
- Create: `utils/kv_cache_reconstruction.py`
- Modify: `tests/test_cage_metrics.py`
- Create: `tests/test_kv_cache_reconstruction.py`

**Interfaces:**
- Produces: `reconstruct_kivi_cache(past_key_value, group_size, bits) -> tuple[Tensor, Tensor]`.
- Produces: `reconstruct_cage_cache(past_key_value) -> tuple[Tensor, Tensor]`.
- Extends: `compute_cage_perturbation_metrics` with three `joint_*` float fields while preserving its existing typed arguments.

- [ ] **Step 1: Write failing joint-metric tests**

```python
def test_joint_metrics_are_zero_for_identical_cache(self):
    inputs = self._small_inputs()
    metrics = compute_cage_perturbation_metrics(**inputs)
    self.assertEqual(metrics["joint_attention_output_mse"], 0.0)
    self.assertEqual(metrics["joint_post_o_proj_mse"], 0.0)
    self.assertEqual(metrics["joint_attention_output_relative_error"], 0.0)

def test_joint_metrics_respond_to_key_and_value_error(self):
    inputs = self._small_inputs()
    inputs["key_states_hat"] = inputs["key_states_hat"] + 0.25
    inputs["value_states_hat"] = inputs["value_states_hat"] - 0.25
    metrics = compute_cage_perturbation_metrics(**inputs)
    self.assertGreater(metrics["joint_attention_output_mse"], 0.0)
    self.assertGreater(metrics["joint_post_o_proj_mse"], 0.0)
```

- [ ] **Step 2: Run and verify missing-field failures**

Run: `python -m unittest tests.test_cage_metrics -v`

Expected: FAIL with missing `joint_attention_output_mse`.

- [ ] **Step 3: Implement the joint path**

Inside `compute_cage_perturbation_metrics`, add:

```python
joint_output = torch.matmul(
    perturbed_scores.to(repeated_values_hat.dtype),
    repeated_values_hat,
)
joint_delta = reference_value_output - joint_output
```

and return:

```python
"joint_attention_output_mse": _as_float(F.mse_loss(reference_value_output, joint_output)),
"joint_post_o_proj_mse": _as_float(_post_o_proj_mse(joint_delta, o_proj_weight)),
"joint_attention_output_relative_error": _as_float(
    _relative_l2_error(reference_value_output, joint_output)
),
```

- [ ] **Step 4: Write failing reconstruction tests**

Construct a 16-token, 16-channel zero-code KIVI tuple with unit scales and a one-token FP16 suffix; construct an equivalent CAGE cache. Assert both reconstruct to shape `[1,1,17,16]`, with zero prefix and one suffix.

- [ ] **Step 5: Implement neutral reconstruction helpers**

```python
def reconstruct_kivi_cache(past_key_value, group_size: int, bits: int):
    from quant.new_pack import unpack_and_dequant_vcache
    k_code, k_full, k_scale, k_min, v_code, v_full, v_scale, v_min, _ = past_key_value
    k_quant = None
    if k_code is not None:
        k_quant = unpack_and_dequant_vcache(
            k_code, k_scale.unsqueeze(-1), k_min.unsqueeze(-1), group_size, bits
        ).transpose(2, 3).contiguous()
    v_quant = None
    if v_code is not None:
        v_quant = unpack_and_dequant_vcache(
            v_code, v_scale.unsqueeze(-1), v_min.unsqueeze(-1), group_size, bits
        )
    return _concat_history(k_quant, k_full), _concat_history(v_quant, v_full)

def reconstruct_cage_cache(past_key_value):
    unpacked = unpack_cage_past_key_value(past_key_value)
    key = _scatter_buckets(
        unpacked.key_cache.key_quant_buckets,
        unpacked.key_cache.key_bucket_indices,
    )
    value = _scatter_buckets(
        unpacked.value_cache.value_quant_buckets,
        unpacked.value_cache.value_bucket_indices,
    )
    return (
        _concat_history(key, unpacked.key_cache.key_full),
        _concat_history(value, unpacked.value_cache.value_full),
    )
```

Implement `_concat_history` with explicit `None` handling and `_scatter_buckets` with a per-head `scatter_` matching `LlamaAttention_KIVI._reconstruct_cage_states`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m unittest tests.test_cage_metrics tests.test_kv_cache_reconstruction -v`

Expected: PASS.

```bash
git add utils/cage_metrics.py utils/kv_cache_reconstruction.py tests/test_cage_metrics.py tests/test_kv_cache_reconstruction.py
git commit -m "feat: add joint KV perturbation metrics"
```

---

### Task 3: Capture the Shared FP16 Query and Candidate Histories

**Files:**
- Modify: `models/llama_kivi.py:45`
- Create: `utils/cage_experiment_hooks.py`
- Modify: `tests/test_cage_attention_integration.py`
- Create: `tests/test_cage_experiment_hooks.py`

**Interfaces:**
- Produces: `begin_reference_capture(model)`, `begin_candidate_capture(model)`, `collect_layer_metrics(model)`, and `reset_experiment_capture(model)`.
- Attention modules expose `set_kv_experiment_phase(phase)` and `pop_kv_experiment_metrics()`.

- [ ] **Step 1: Write failing phase/capture tests**

Use the one-layer test attention to run reference T+1, candidate prefill T, and one decode token. Assert the captured reference query has shape `[1,H,1,D]`, `query_source` is `fp16_reference_final_position`, and the metric record contains all joint fields.

- [ ] **Step 2: Verify the API is absent**

Run: `python -m unittest tests.test_cage_attention_integration tests.test_cage_experiment_hooks -v`

Expected: FAIL with missing `set_kv_experiment_phase` or `begin_reference_capture`.

- [ ] **Step 3: Add attention-local experiment state**

Initialize these fields in `LlamaAttention_KIVI.__init__`:

```python
self._kv_experiment_phase = "off"
self._kv_reference_query = None
self._kv_reference_key_history = None
self._kv_reference_value_history = None
self._kv_key_importance = None
self._kv_value_importance = None
self._kv_experiment_metrics = None
```

Add phase control:

```python
def set_kv_experiment_phase(self, phase: str) -> None:
    if phase not in {"off", "reference", "candidate"}:
        raise ValueError(f"unsupported KV experiment phase: {phase}")
    if phase == "reference":
        self._kv_reference_query = None
        self._kv_reference_key_history = None
        self._kv_reference_value_history = None
        self._kv_key_importance = None
        self._kv_value_importance = None
        self._kv_experiment_metrics = None
    elif phase == "candidate" and self._kv_reference_query is None:
        raise RuntimeError("reference query must be captured before candidate phase")
    self._kv_experiment_phase = phase

def pop_kv_experiment_metrics(self):
    metrics = self._kv_experiment_metrics
    self._kv_experiment_metrics = None
    return metrics
```

Immediately after rotary embedding, reference phase stores `query_states[:, :, -1:, :].detach()`. Candidate prefill stores detached FP16 Key/Value history and computes/stores Key/Value importance. Candidate decode calls the neutral metric helper with the stored FP16 query; concatenate the current exact K/V to both the FP16 history and reconstructed history before computing attention metrics.

For KIVI, reconstruct the incoming `past_key_value` before its decode branch mutates buffers. For CAGE, use the reconstructed previous states already available in `_cage_fake_forward`. Set metadata fields `phase=teacher_forced_decode` and `query_source=fp16_reference_final_position`.

- [ ] **Step 4: Implement model-level hooks**

```python
def _attentions(model):
    return [layer.self_attn for layer in model.model.layers]

def begin_reference_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("reference")

def begin_candidate_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("candidate")

def collect_layer_metrics(model):
    records = []
    for layer_index, attention in enumerate(_attentions(model)):
        metrics = attention.pop_kv_experiment_metrics()
        if metrics is None:
            raise RuntimeError(f"layer {layer_index} did not emit KV metrics")
        records.append({"layer_index": layer_index, **metrics})
    return records

def reset_experiment_capture(model):
    for attention in _attentions(model):
        attention.set_kv_experiment_phase("off")
        attention._kv_reference_query = None
        attention._kv_reference_key_history = None
        attention._kv_reference_value_history = None
        attention._kv_key_importance = None
        attention._kv_value_importance = None
        attention._kv_experiment_metrics = None
```

Test that reset clears every retained tensor before the next sample-length point.

- [ ] **Step 5: Run integration tests and commit**

Run: `python -m unittest tests.test_cage_attention_integration tests.test_cage_experiment_hooks -v`

Expected: PASS.

```bash
git add models/llama_kivi.py utils/cage_experiment_hooks.py tests/test_cage_attention_integration.py tests/test_cage_experiment_hooks.py
git commit -m "feat: capture shared FP16 KV metric reference"
```

---

### Task 4: Separate Paper and Runtime Memory Namespaces

**Files:**
- Modify: `utils/cage_memory.py`
- Modify: `tests/test_cage_memory.py`

**Interfaces:**
- Preserves: `summarize_cache_bytes(cache)` as paper-facing backward-compatible summary.
- Produces: `summarize_runtime_cache_bytes(cache) -> dict[str,int|str]`.
- Produces: `summarize_fp16_cache_bytes((key,value)) -> dict[str,int|str]`.
- Produces: `sum_cache_summaries(layer_summaries) -> dict[str,int|str]`.

- [ ] **Step 1: Write failing namespace and aggregation tests**

```python
def test_cage_runtime_summary_counts_fake_tensors(self):
    cache = self._cage_cache()
    paper = summarize_cache_bytes(cache)
    runtime = summarize_runtime_cache_bytes(cache)
    self.assertGreater(runtime["total_bytes"], paper["payload_only_bytes"])
    self.assertEqual(runtime["cache_type"], "cage_fake_runtime")

def test_sum_cache_summaries_adds_every_byte_field(self):
    total = sum_cache_summaries([
        {"total_bytes": 10, "payload_only_bytes": 4},
        {"total_bytes": 20, "payload_only_bytes": 8},
    ])
    self.assertEqual(total["total_bytes"], 30)
    self.assertEqual(total["payload_only_bytes"], 12)
```

- [ ] **Step 2: Run and verify missing helpers**

Run: `python -m unittest tests.test_cage_memory -v`

Expected: FAIL on imports.

- [ ] **Step 3: Implement runtime, FP16, and model-wide summaries**

`summarize_runtime_cache_bytes` must count actual tensor `nbytes` for every CAGE fake bucket, residual, and index; for KIVI it may reuse actual packed tensor sizes. `summarize_fp16_cache_bytes` counts Key and Value tensor bytes as residual/full-precision bytes with zero payload metadata. `sum_cache_summaries` sums every numeric field present and sets `cache_type="model_total"`.

Add `metadata_bytes = key_scale_bytes + value_scale_bytes + key_min_or_zp_bytes + value_min_or_zp_bytes` to every paper summary. Keep bucket indices separate from metadata so plotting code cannot silently hide their cost.

Return the worker namespace exactly as:

```python
memory = {
    "paper_estimate": paper_summary,
    "runtime_tensors": runtime_summary,
    "cuda_peak_diagnostic": {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    },
}
```

- [ ] **Step 4: Run and commit**

Run: `python -m unittest tests.test_cage_memory -v`

Expected: PASS.

```bash
git add utils/cage_memory.py tests/test_cage_memory.py
git commit -m "feat: separate paper and runtime cache memory"
```

---

### Task 5: Define and Validate the Experiment Manifest

**Files:**
- Create: `utils/cage_experiment_config.py`
- Create: `tests/test_cage_experiment_config.py`

**Interfaces:**
- Produces: `load_and_resolve_manifest(path: str|Path) -> dict`.
- Produces: `expand_jobs(manifest: dict) -> list[dict]`.

- [ ] **Step 1: Write failing validation and expansion tests**

Use a temporary manifest with FP16, one valid KIVI pair, and one CAGE config. Assert three jobs. Add failures for unknown method, prompt length above `max_position_embeddings`, and KIVI `residual_length % group_size != 0`.

- [ ] **Step 2: Run and verify module absence**

Run: `python -m unittest tests.test_cage_experiment_config -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement strict resolution**

Use this resolved job shape:

```python
{
    "job_id": "kivi-g32-r32",
    "method": "kivi",
    "model": {"reference": "/root/autodl-tmp/models/Llama-2-7b-hf", "dtype": "float16", "device": "cuda", "max_position_embeddings": 4096},
    "method_config": {"k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 32},
    "sample_ids": ["doc-001", "doc-002", "doc-003"],
    "prompt_lengths": [512, 1024, 2048, 4096],
    "measurement": {"decode_tokens": 1, "seed": 0},
    "prompts_file": "/root/autodl-tmp/cage_pilot_prompts.jsonl",
    "output_dir": "/root/autodl-tmp/cage_pareto_pilot",
}
```

Reject unknown top-level or method fields, non-unique IDs, non-positive lengths, `decode_tokens != 1`, unsupported dtype/device, unsupported methods, and invalid KIVI pairs. Resolve all CAGE defaults explicitly, including bucket groups and clip percentiles.

- [ ] **Step 4: Run and commit**

Run: `python -m unittest tests.test_cage_experiment_config -v`

Expected: PASS.

```bash
git add utils/cage_experiment_config.py tests/test_cage_experiment_config.py
git commit -m "feat: validate CAGE experiment manifests"
```

---

### Task 6: Add Prompt, Provenance, Run-ID, and Atomic Output Utilities

**Files:**
- Create: `utils/cage_experiment_io.py`
- Create: `tests/test_cage_experiment_io.py`

**Interfaces:**
- Produces: `load_prompt_records(path) -> dict[str,PromptRecord]`.
- Produces: `prepare_prompt(tokenizer, text, prompt_length) -> dict[str,Tensor|str|int]`.
- Produces: `source_state_identity(repo_root) -> dict`, `collect_provenance(repo_root) -> dict`, and `stable_run_id(payload) -> str`.
- Produces: `atomic_write_json`, `atomic_write_jsonl`, and `aggregate_completed_runs`.

- [ ] **Step 1: Write failing utility tests**

Test duplicate/missing sample IDs, insufficient T+1 tokens, deterministic truncation, stable IDs under reordered dictionaries, provenance keys, atomic replacement, exclusion of failed runs, and CSV flattening.

- [ ] **Step 2: Run and verify module absence**

Run: `python -m unittest tests.test_cage_experiment_io -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement exact prompt slicing and stable IDs**

```python
def prepare_prompt(tokenizer, text: str, prompt_length: int):
    encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    if encoded.shape[-1] < prompt_length + 1:
        raise ValueError(f"text has {encoded.shape[-1]} tokens; need {prompt_length + 1}")
    return {
        "prompt_ids": encoded[:, :prompt_length].contiguous(),
        "continuation_ids": encoded[:, prompt_length:prompt_length + 1].contiguous(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "effective_prompt_length": prompt_length,
    }

def stable_run_id(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
```

`source_state_identity` records `git rev-parse HEAD`; when dirty, hash `git diff --binary HEAD` plus sorted untracked experiment-code paths and contents. Atomic writers create a temporary file in the destination directory, flush and `os.fsync`, then call `os.replace`.

`collect_provenance` records source identity, dirty flag, Python, PyTorch, Transformers, CUDA runtime, CUDA driver, GPU name, deterministic seed, and normalized command arguments. CUDA-only values are `None` on CPU tests rather than omitted.

- [ ] **Step 4: Run and commit**

Run: `python -m unittest tests.test_cage_experiment_io -v`

Expected: PASS.

```bash
git add utils/cage_experiment_io.py tests/test_cage_experiment_io.py
git commit -m "feat: add reproducible experiment IO"
```

---

### Task 7: Implement the Single-Configuration Worker

**Files:**
- Create: `scripts/cage_experiment_worker.py`
- Create: `tests/test_cage_experiment_worker.py`

**Interfaces:**
- CLI: `python scripts/cage_experiment_worker.py --manifest <resolved.json> --job-index <N>`.
- Consumes Tasks 2-6; writes authoritative run/layer/failure files.

- [ ] **Step 1: Write failing import-safe worker tests**

Load the script with `importlib`, test argument parsing, method config application, layer aggregation, FP16 zero metric records, and skip behavior when a completed run file exists. Use fake tokenizer/model objects; do not require CUDA.

- [ ] **Step 2: Run and verify script absence**

Run: `python -m unittest tests.test_cage_experiment_worker -v`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement model loading and method configuration**

Load the tokenizer and validate every selected prompt has T+1 tokens before loading model weights. For `fp16`, load `transformers.LlamaForCausalLM`. For `kivi` and `cage`, load `LlamaForCausalLM_KIVI`. Apply all resolved values before `from_pretrained`; set `cage_enable=False` for KIVI and `cage_enable=True`, `cage_mode="fake"` for CAGE. Require `model_type == "llama"`.

- [ ] **Step 4: Implement one point**

Candidate method pseudocode must be realized exactly in this order:

```python
begin_reference_capture(model)
with torch.inference_mode():
    model(input_ids=full_t_plus_one_ids, use_cache=False, return_dict=True)
begin_candidate_capture(model)
with torch.inference_mode():
    prefill = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    model(
        input_ids=continuation_ids,
        past_key_values=prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
layer_metrics = collect_layer_metrics(model)
```

Summarize memory from `prefill.past_key_values`, so the memory axis is exactly T tokens. FP16 emits zero metric dictionaries with the same schema. Record load, reference, prefill, decode, and metric durations as diagnostics only.

Build `run_id` from resolved job config, sample ID/text hash, prompt length, model identity, and source-state identity. Write `layers/<run_id>.jsonl` before atomically writing completed `runs/<run_id>.json`. On exception write `failures/<run_id>.json` with category and stage, then continue to the next point unless model state is unusable.

Failure records use `{"schema_version": 1, "run_id": run_id, "status": "failed", "category": category, "stage": stage, "retryable": retryable, "message": str(error)}` and never appear in completed summaries.

Use this authoritative record shape; fill every nested value from resolved configuration or measured data:

```python
run_record = {
    "schema_version": 1,
    "run_id": run_id,
    "status": "completed",
    "model": model_record,
    "method": {"name": job["method"], "resolved_config": job["method_config"]},
    "input": input_record,
    "quantization": quantization_record,
    "measurement": {
        "phase": "teacher_forced_decode",
        "query_source": "fp16_reference_final_position",
        "query_count": 1,
        "layer_count": len(layer_records),
    },
    "memory": memory_record,
    "metrics_aggregate": aggregate_layer_metrics(layer_records),
    "runtime_diagnostics": timing_record,
    "provenance": provenance_record,
}
```

`aggregate_layer_metrics` collects each numeric metric across layers and returns `{"mean": float(statistics.fmean(values)), "median": float(statistics.median(values)), "max": float(max(values))}` for that metric. In a `finally` block call `reset_experiment_capture(model)`, delete point-local outputs and caches, run `gc.collect()`, and call `torch.cuda.empty_cache()` on CUDA.

- [ ] **Step 5: Run worker tests and full CPU suite**

Run: `python -m unittest tests.test_cage_experiment_worker -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cage_experiment_worker.py tests/test_cage_experiment_worker.py
git commit -m "feat: add CAGE experiment worker"
```

---

### Task 8: Implement Matrix Orchestration, Resume, and Aggregation

**Files:**
- Create: `scripts/cage_run_matrix.py`
- Create: `tests/test_cage_run_matrix.py`

**Interfaces:**
- CLI: `python scripts/cage_run_matrix.py --manifest <manifest.json> [--retry-transient-once]`.
- Produces: resolved manifest, subprocess jobs, canonical JSONL summary, flattened CSV.

- [ ] **Step 1: Write failing orchestrator tests**

Mock `subprocess.run`. Assert one command per resolved job, nonzero deterministic failures are not retried, one transient process failure is retried once when enabled, completed points remain resumable, and aggregation ignores failures.

- [ ] **Step 2: Run and verify script absence**

Run: `python -m unittest tests.test_cage_run_matrix -v`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement orchestration**

```python
manifest = load_and_resolve_manifest(args.manifest)
jobs = expand_jobs(manifest)
atomic_write_json(output_dir / "manifest.resolved.json", {**manifest, "jobs": jobs})
for job_index, _ in enumerate(jobs):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "cage_experiment_worker.py"),
        "--manifest", str(output_dir / "manifest.resolved.json"),
        "--job-index", str(job_index),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        handle_job_failure(completed.returncode, command, args.retry_transient_once)
aggregate_completed_runs(output_dir)
```

Classify worker exit codes: 0 success, 2 manifest/input error, 3 deterministic CUDA OOM, 4 model-load error, 5 transient process error. Retry only code 5, once.

- [ ] **Step 4: Run tests and commit**

Run: `python -m unittest tests.test_cage_run_matrix -v`

Expected: PASS.

```bash
git add scripts/cage_run_matrix.py tests/test_cage_run_matrix.py
git commit -m "feat: orchestrate resumable CAGE experiments"
```

---

### Task 9: Add the Pilot Manifest, Documentation, and Acceptance Commands

**Files:**
- Create: `configs/cage_pilot_llama2_7b.json`
- Create: `configs/cage_pilot_llama2_7b_acceptance.json`
- Create: `docs/cage_experiments.md`
- Modify: `README.md`

**Interfaces:**
- Produces the exact 120-point pilot and documented reduced acceptance invocation.

- [ ] **Step 1: Add and validate the exact manifest**

The manifest must use three named samples, four prompt lengths, FP16, these KIVI pairs:

```json
[[32,32],[32,64],[32,128],[64,64],[64,128],[128,128]]
```

and CAGE residuals `[32,64,128]` with group profile `[32,64,128]` and clips `[0.999,0.995,0.99]`. Use `/root/autodl-tmp/cage_pilot_prompts.jsonl` and `/root/autodl-tmp/cage_pareto_pilot` as the documented server paths.

Run: `python -m unittest tests.test_cage_experiment_config -v`

Expected: PASS and `len(expand_jobs(manifest)) == 10` (1 FP16 + 6 KIVI + 3 CAGE).

- [ ] **Step 2: Document preparation and interpretation**

Document the input JSONL contract, the full command:

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b.json --retry-transient-once
```

the reduced acceptance matrix, output schema, resume behavior, and the prohibition on treating fake runtime memory/timing as compressed-kernel results.

- [ ] **Step 3: Run final CPU verification**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

Run: `python scripts/cage_experiment_worker.py --help`

Expected: exit 0 and show `--manifest` and `--job-index`.

Run: `python scripts/cage_run_matrix.py --help`

Expected: exit 0 and show `--manifest`.

- [ ] **Step 4: Run the required Llama-2-7B GPU acceptance on AutoDL**

Run the checked-in acceptance manifest, which contains sample `doc-001`, prompt lengths 512 and 2048, FP16, KIVI `(32,32)`, and CAGE default `R=32`:

```bash
python scripts/cage_run_matrix.py --manifest configs/cage_pilot_llama2_7b_acceptance.json --retry-transient-once
```

Expected:

- six completed run records and no failure records;
- 32 layer records per run;
- finite non-negative metrics;
- FP16 joint errors equal zero within numerical tolerance;
- Key/Value cache lengths equal T;
- Key residual length equals `T % R`, Value residual equals `min(T,R)`;
- model-wide bytes equal the sum of 32 layers;
- a second identical run skips all six points.

- [ ] **Step 5: Commit**

```bash
git add configs/cage_pilot_llama2_7b.json configs/cage_pilot_llama2_7b_acceptance.json docs/cage_experiments.md README.md
git commit -m "docs: add CAGE Pareto pilot workflow"
```

---

## Final Verification Gate

- [ ] Run `python -m unittest discover -s tests -v` and retain the complete passing output.
- [ ] Run `git status --short` and confirm only intentionally uncommitted server result artifacts, if any, remain.
- [ ] Compare `manifest.resolved.json` against the design matrix: exactly 10 jobs and 120 full-pilot points.
- [ ] Check every completed run has one authoritative JSON record, 32 layer JSONL records, and matching schema version.
- [ ] Check paper, runtime tensor, and CUDA diagnostic memory namespaces are distinct in JSON and CSV.
- [ ] Check the README and experiment documentation make no latency, throughput, downstream-quality, or realized-compression claims.
