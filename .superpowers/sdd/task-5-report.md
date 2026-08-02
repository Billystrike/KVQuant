# Task 5 Report: Experiment Manifest

## Status

Implemented strict JSON manifest loading/resolution and deterministic method-to-job expansion.

## Changes

- Added `utils/cage_experiment_config.py` with `load_and_resolve_manifest` and `expand_jobs`.
- Rejects unknown nested/top-level fields, missing fields, duplicate method/sample IDs, invalid lengths/context limits, unsupported methods/dtypes/devices, non-one-token decode protocols, invalid bit widths, malformed CAGE buckets, and invalid KIVI group/residual pairs.
- Resolves every CAGE method default explicitly, including Key/Value bucket group sizes and clipping percentiles.
- Added focused unit tests covering three-method expansion and required rejection cases.

## TDD Evidence

- RED: `python -m unittest tests.test_cage_experiment_config -v` failed with the expected `ModuleNotFoundError: No module named 'utils.cage_experiment_config'`.
- GREEN: the same focused command passed 5 tests.

## Verification

- `python -m unittest tests.test_cage_experiment_config -v`: 5 tests passed.
- `python -m unittest discover -s tests -v`: 67 tests passed.
- `git diff --check`: passed.

The full suite emits one pre-existing Transformers `TRANSFORMERS_CACHE` deprecation warning; it has no test impact.

## Review Fix Evidence

- Added pre-membership scalar type validation for model dtype/device, method names, and K/V bit widths so malformed list/dict JSON values consistently raise `ValueError` instead of leaking `TypeError`.
- Added regression coverage for unhashable scalar inputs, nested unknown/missing model and measurement fields, and invalid CAGE bucket group/count/clip values.
- RED: the focused unhashable-scalar test produced six `TypeError: unhashable type` errors before the implementation change.
- GREEN: `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest tests.test_cage_experiment_config -v` passed 8 tests.
- Full verification: `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest discover -s tests -v` passed 70 tests. The existing Transformers cache deprecation warning remains non-fatal.
