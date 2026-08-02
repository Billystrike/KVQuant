# Task 3 Report: Shared FP16 Query and Candidate Histories

## Status

Implemented and verified.

## Changes

- Added attention-local experiment phases and retained state to `LlamaAttention_KIVI`.
- Captured the post-RoPE final-position query from the T+1 FP16 reference pass.
- Captured detached candidate-prefill K/V histories and key/value importance.
- Computed teacher-forced decode metrics from the shared reference query for both KIVI and CAGE.
- Appended the current exact K/V token to both exact and reconstructed histories before metric computation.
- Reconstructed KIVI cache state before decode buffer mutation, only while candidate capture is active.
- Added model-level begin, collect, and reset hooks.
- Added integration and hook tests, including 4K retained-tensor reset and capture-off regression coverage.

## TDD Evidence

- Initial focused run failed because `utils.cage_experiment_hooks` was absent.
- After the hook module was added, the focused run failed because `set_kv_experiment_phase` was absent.
- The KIVI shared-query test failed because decode emitted no metric record before KIVI wiring was added.
- The capture-off regression test failed because KIVI reconstructed its cache unconditionally; the implementation was then gated to candidate phase.

## Verification

- Focused: `python -m unittest tests.test_cage_attention_integration tests.test_cage_experiment_hooks -v` — 10 passed.
- Full: `python -m unittest discover -s tests -v` — 56 passed.
- Syntax: `python -m py_compile models/llama_kivi.py utils/cage_experiment_hooks.py tests/test_cage_attention_integration.py tests/test_cage_experiment_hooks.py` — passed.
- Whitespace: `git diff --check` — passed (Git emitted only existing LF-to-CRLF conversion warnings).

## Review

Independent static review found no Critical, Important, or Minor issues. Manual review additionally caught and fixed the capture-off KIVI reconstruction overhead before final verification.

## Concerns

- The configured environment emits the existing Transformers `TRANSFORMERS_CACHE` deprecation warning; it does not affect test results.

## Review Fix: Separate KIVI Key/Value Bit Widths

- Changed `reconstruct_kivi_cache` to accept `k_bits` and `v_bits` separately and use them for the corresponding packed histories.
- Updated both Llama KIVI reconstruction call sites and all direct test callers.
- Added a mismatched-width regression test that verifies Key uses 2 bits while Value uses 4 bits.

### TDD Evidence

- RED: `python -m unittest tests.test_kv_cache_reconstruction.KvCacheReconstructionTest.test_kivi_reconstruction_uses_separate_key_and_value_widths -v` failed with `TypeError: reconstruct_kivi_cache() got an unexpected keyword argument 'k_bits'`.
- GREEN focused: `python -m unittest tests.test_kv_cache_reconstruction tests.test_cage_attention_integration tests.test_cage_experiment_hooks -v` — 16 passed.

### Verification

- Full: `python -m unittest discover -s tests -v` — 57 passed.
- The existing Transformers `TRANSFORMERS_CACHE` deprecation warning remains the only observed warning.
