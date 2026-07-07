# Task 09: MVP Tests and Smoke Scripts

## Objective

Provide minimal tests and runnable smoke checks for the first CAGE-KV deliverable.

## Unit tests

Create or extend tests for:

1. Config normalization.
2. Key importance shapes and non-negativity.
3. Value importance shapes and non-negativity.
4. Bucket assignment coverage and determinism.
5. Key fake quant output shape and scatter correctness.
6. Value fake quant output shape and scatter correctness.
7. Memory summary JSON serializability.
8. Metric dictionary JSON serializability.

## Suggested test file

- `quant/test_cage_quant.py` for quantization and bucket tests.
- `tests/test_cage_config.py` if a tests directory exists or is introduced.

## Smoke script

Create `scripts/cage_smoke.py`.

The script should:

1. Load a small model or support a user-provided model path.
2. Run a short prompt with original KIVI settings.
3. Run the same short prompt with `cage_enable=True` and `cage_mode="fake"`.
4. Generate 8 to 16 tokens.
5. Print cache memory summary and optional perturbation metrics.

## Acceptance criteria

- Unit tests pass on CPU where possible.
- Smoke script documents GPU/model requirements clearly.
- CAGE disabled and CAGE fake modes both produce the expected number of generated tokens.
- Failures include actionable error messages for unsupported environment or missing model files.
