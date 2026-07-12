# Task 7 Report: Single-Configuration Worker

## Status

Complete. Added the import-safe single-job worker and focused CPU tests.

## Implementation

- Validates all selected T+1 prompt inputs with the tokenizer before model weights load.
- Uses HF `LlamaForCausalLM` for FP16 and the custom `LlamaForCausalLM_KIVI` for KIVI/CAGE.
- Applies resolved configuration before `from_pretrained`, including KIVI/CAGE capture settings.
- Executes reference, T-token prefill, one teacher-forced decode, layer collection, and exact three-namespace memory summaries.
- Writes atomic layer records before completed run records, structured failures on load/point errors, stable IDs, and skips all completed points before loading weights.
- Performs point cleanup and CUDA cache cleanup; FP16 emits the identical numeric metric schema with zero values.

## TDD Evidence

- Initial focused run failed with `FileNotFoundError` because the worker script did not exist.
- Added tests for CLI parsing, method configuration, aggregation, FP16 metric schema, completed-record detection, and full-job skip-before-weight-loading.
- Focused verification: 6/6 passed.
- Full CPU verification: 88/88 passed.
- `py_compile` and `git diff --check`: passed.

## Review

Independent review identified pre-skip model loading, missing layer identity/memory fields, and fragile cleanup. These were addressed before commit. The worker now computes pending points first, emits structured load failures, enriches each layer record, and cleans point-local state in `finally`.

## Concerns

- The local environment has Transformers 4.57.3 rather than the target 4.43.1. The implementation avoids 4.57-only APIs and uses the legacy tuple cache contract expected by 4.43.1, but exact-version runtime verification was not available locally.
- CPU tests use fakes/helpers and do not load real Llama weights or exercise KIVI CUDA/Triton kernels; GPU/server validation remains necessary.
- The suite emits the pre-existing `TRANSFORMERS_CACHE` deprecation warning and a sandbox-related Git global-ignore warning.

## Review Fix TDD Evidence

- RED: focused worker suite failed 1 assertion and 3 missing-helper errors: metadata leaked into `metrics_aggregate`, and lifecycle, layer-count validation, and stale-failure cleanup behavior were absent.
- GREEN: focused worker suite passed 9/9 with the target Python environment.
- Regression: full CPU suite passed 91/91 with the target Python environment.
- The worker now aggregates only `METRIC_NAMES`, releases the reference forward output and runs collection before CUDA peak reset/prefill, rejects metric/memory layer-count mismatches before writes, and removes a stale failure record only after both authoritative success files are written.

## Exit-Code Contract Review Fix

- Added a shared classifier covering every documented worker failure outcome:
  input 2, CUDA OOM 3, non-OOM load 4, and transient/unusable runtime 5.
- Load and point failures use the classifier; usable point failures retain
  structured records and continuation, while capture-reset failure stops use
  of the now-unusable model with code 5.
- A source scan verifies there are no explicit code-1 worker exits.
