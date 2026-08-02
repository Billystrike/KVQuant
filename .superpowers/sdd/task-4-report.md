# Task 4 Report: Separate Paper and Runtime Memory Namespaces

## Status

Complete.

## Commit

`d3cfac6 feat: separate paper and runtime cache memory`

## Implemented

- Preserved `summarize_cache_bytes` as the paper-facing packed-memory summary.
- Added `metadata_bytes` to finalized summaries as the sum of scale and min/zero-point bytes; bucket indices remain a separate field.
- Added `summarize_runtime_cache_bytes` for actual CAGE fake-runtime tensor storage and actual KIVI tensor storage.
- Added `summarize_fp16_cache_bytes` with key/value tensors counted as full-precision residual bytes.
- Added `sum_cache_summaries`, which sums every numeric field present and labels the result `model_total`.
- Added focused tests for namespace separation, actual fake tensor bytes, FP16 accounting, aggregation, and metadata/index separation.

## TDD Evidence

RED: `python -m unittest tests.test_cage_memory -v` failed on the expected missing `sum_cache_summaries` import before production implementation.

GREEN: the focused suite passed 7 tests after the minimal implementation.

## Verification

- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest tests.test_cage_memory -v`: PASS, 7 tests.
- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest discover -s tests -v`: PASS, 61 tests.
- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m py_compile utils\cage_memory.py tests\test_cage_memory.py`: PASS.
- `git diff --check`: PASS.

## Self-review

- Runtime CAGE payload fields use each fake quant bucket's real tensor `nbytes`, not packed INT2 estimates.
- Runtime residuals and bucket indices use their real tensor `nbytes` and remain separately visible.
- KIVI runtime accounting reuses the existing actual packed tensor accounting.
- FP16 metadata and payload fields remain zero; key/value storage is represented in `residual_full_precision_bytes`.
- Aggregation does not assume a fixed byte-field list and therefore preserves numeric fields introduced by callers.

## Concerns

- The full suite emits the pre-existing Transformers `TRANSFORMERS_CACHE` deprecation warning; it does not affect test results.
- The worker-level `memory = {paper_estimate, runtime_tensors, cuda_peak_diagnostic}` assembly is not in the two Task 4 files and was not modified as part of this scoped task.

## Review Fix

- Added the typed `build_memory_namespace` production helper for Task 7. It produces exactly `paper_estimate`, `runtime_tensors`, and `cuda_peak_diagnostic`, accepting already-collected CUDA allocated/reserved peak byte values.
- Replaced the value-dependent metadata/index inequality assertion with independent checks that both fields exist, `metadata_bytes` uses the required formula, and `total_bytes` accounts for metadata and bucket indices separately.

### Review-fix TDD evidence

- RED: `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest tests.test_cage_memory -v` failed with the expected `ImportError: cannot import name 'build_memory_namespace'` before production implementation.
- GREEN: the same focused command passed 8 tests after the minimal helper implementation.

### Review-fix verification

- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest tests.test_cage_memory -v`: PASS, 8 tests.
- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest discover -s tests -v`: PASS, 62 tests.
- `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m py_compile utils\cage_memory.py tests\test_cage_memory.py`: PASS.
- `git diff --check`: PASS.
