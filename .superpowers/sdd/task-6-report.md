# Task 6 Report

## Status

Complete. Implemented prompt JSONL validation, exact T/T+1 token slicing, canonical run IDs, Git source-state identity, provenance capture, atomic JSON/JSONL output, and completed-run JSONL/CSV aggregation.

## Commit

`e55880b feat: add reproducible experiment IO`

## TDD evidence

- Initial focused test run failed with `ModuleNotFoundError: No module named 'utils.cage_experiment_io'`.
- The source-state regression test failed while an untracked output JSON incorrectly marked the repository dirty; it passed after restricting untracked identity inputs to experiment-code file types.
- Focused: `python -m unittest tests.test_cage_experiment_io -v` — 7 tests passed.
- Full CPU suite: `python -m unittest discover -s tests -v` — 77 tests passed.
- `git diff --check` completed without errors before commit.

## Implementation notes

- Prompt records accept exactly `sample_id` and `text`, both non-empty strings, and reject duplicate identifiers, malformed JSON, missing fields, unknown fields, and empty files.
- Source identity includes the HEAD commit, tracked binary diff, and sorted untracked experiment-code paths/content. Output data such as JSON result files does not perturb the identity.
- CPU provenance retains `cuda_runtime`, `cuda_driver`, and `gpu_name` with explicit `None` values.
- Atomic writers use a temporary file in the destination directory, flush, `fsync`, and `os.replace`, cleaning the temporary file on failure.
- Aggregation reads only `runs/*.json` records whose status is `completed`, sorts by run ID, and rebuilds canonical JSONL plus dotted-key CSV.

## Concerns

- Test output contains an existing Transformers `TRANSFORMERS_CACHE` deprecation warning.
- Temporary Git-repository tests emit a Windows Git warning because the sandbox cannot read the user-level global ignore file; this does not affect results.

## Review follow-up (2026-07-12)

- Added the public `normalize_command_arguments(command_args, repo_root)` contract. It resolves values for `--manifest`, `--output-dir`, and `--prompts-file` against the repository root, collapses dot segments, emits forward-slash absolute paths, supports both CLI assignment forms, and preserves all non-path arguments.
- Updated `collect_provenance` to store the normalized command.
- Added focused malformed-JSONL regressions for invalid JSON, non-object values, unknown fields, and empty identifiers/text.
- Added aggregation regressions for union CSV columns and the zero-completed case; empty aggregations now produce empty canonical JSONL and CSV files.
- TDD red: the focused suite initially failed because `normalize_command_arguments` did not exist.
- Focused verification: `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest tests.test_cage_experiment_io -v` — 12 tests passed.
- Full CPU verification: `D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe -m unittest discover -s tests -v` — 82 tests passed.
