# Task 01: Configuration and Feature Gate

## Objective

Add a safe CAGE-KV configuration surface so future code can enable the MVP without changing default KIVI behavior.

## Scope

Introduce dynamic config attributes used by `models/llama_kivi.py` first, then mirror them in `models/mistral_kivi.py` only after the Llama path is stable.

## Required config fields

### Core switches

- `config.cage_enable: bool = False`
- `config.cage_mode: str = "fake"`
- `config.cage_k_enable: bool = True`
- `config.cage_v_enable: bool = True`

### Key policy

- `config.cage_k_importance: str = "q2_var"`
- `config.cage_k_group_sizes: list[int] = [32, 64, 128]`
- `config.cage_k_clip_percentiles: list[float] = [0.999, 0.995, 0.99]`
- `config.cage_k_num_buckets: int = 3`
- `config.cage_k_flush_length: int = 128`

### Value policy

- `config.cage_v_importance: str = "wo_var"`
- `config.cage_v_group_sizes: list[int] = [32, 64, 128]`
- `config.cage_v_clip_percentiles: list[float] = [0.999, 0.995, 0.99]`
- `config.cage_v_num_buckets: int = 3`

### Debug and reporting

- `config.cage_collect_metrics: bool = False`
- `config.cage_dump_dir: str | None = None`
- `config.cage_memory_summary: bool = False`

## Design requirements

1. All fields must have safe defaults when missing from a pretrained Transformers config.
2. `cage_enable=False` must preserve the original KIVI code path exactly.
3. `cage_mode="fake"` is the only required MVP mode.
4. Unknown `cage_mode` values should raise a clear `ValueError` only when CAGE is enabled.
5. Group-size and clipping lists must match the requested bucket count.

## Suggested helper

Create `models/cage_config.py` with:

```python
get_cage_config(config) -> CageConfig
```

The helper should normalize missing attributes and validate simple invariants. A small dataclass is preferred for readability.

## Acceptance criteria

- Loading an existing KIVI model without CAGE-specific attributes still works.
- Enabling `config.cage_enable=True` exposes normalized CAGE settings.
- Invalid bucket lengths are caught before generation starts.
- No CUDA code is touched in this task.
