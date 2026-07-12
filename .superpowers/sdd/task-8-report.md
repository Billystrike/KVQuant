# Task 8 Report: Matrix Orchestration, Resume, and Aggregation

## Status

Implemented the import-safe matrix CLI at `scripts/cage_run_matrix.py` with:

- strict manifest loading and deterministic job expansion;
- atomic `manifest.resolved.json` output with portable absolute input/output paths;
- one worker subprocess per resolved job, using the active Python interpreter;
- exact handling of documented worker exit codes 0, 2, 3, 4, and 5;
- a single retry only for exit code 5 when `--retry-transient-once` is enabled;
- continued execution after per-job codes 3, 4, and 5, and early stop for shared manifest/input code 2;
- unconditional completed-run aggregation after documented worker failures;
- preservation of worker-owned completed-point resume behavior.

## TDD Evidence

Initial focused run failed with six `FileNotFoundError` errors because
`scripts/cage_run_matrix.py` did not exist. After the minimal implementation,
the focused suite passed. A relative-path regression test was then added,
observed failing against the first implementation, and fixed by resolving paths
relative to the source manifest before relocating the resolved manifest.

## Verification

- `python -m unittest tests.test_cage_run_matrix -v`: 7 tests passed.
- `python -m unittest discover -s tests -v`: 98 tests passed.
- `git diff --check`: passed.

The configured interpreter was
`D:\A_develop_tool\miniconda\envs\llm-compressor\python.exe`.

## Concerns

The existing Task 7 worker currently returns undocumented exit code 1 for
some model-load or unusable-state paths. The Task 8 orchestrator deliberately
rejects undocumented worker codes rather than guessing whether code 1 means
code 3, 4, or 5. Aligning those Task 7 worker paths with the documented exit
code contract remains necessary for end-to-end classification of real failures.

The test environment also emits an existing Transformers cache deprecation
warning and a Git global-ignore permission warning; neither affected results.

## Commit

Commit message: `feat: orchestrate resumable CAGE experiments`.
