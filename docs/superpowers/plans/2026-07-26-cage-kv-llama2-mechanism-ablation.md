# Llama-2-7B mechanism ablation execution plan

1. Add an explicit `cage_ablation` scientific-identity flag. Core manifests
   continue to reject alternate importance policies.
2. Add deterministic per-layer fixed-random assignment with an explicit base
   seed; retain computed `q2_var`/`wo_var` weights for perturbation metrics.
3. Check in the 24-point acceptance and 144-point full manifests.
4. Run the full CPU suite and server preflight.
5. Execute and validate the 24-point acceptance subset, including exact memory
   equality between full and fixed-random variants.
6. Execute the full matrix, validate 144 runs and 4,608 layer records, and back
   up raw artifacts, resolved manifests, logs, environment, and source state.
7. Generate paired mechanism tables and figures. Preserve all samples and
   lengths; do not filter out adverse contrasts.
8. Apply the frozen advancement rule and write the paired-PPL ablation manifest
   before running it.
