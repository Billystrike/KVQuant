# CAGE-KV cache-conditioned PPL implementation plan

1. Add strict manifest parsing and frozen WikiText-2 corpus validation.
2. Add deterministic anchor selection, nested prompt construction, case IDs,
   completed-case schema validation, and expected-ID aggregation.
3. Add a GPU runner that loads each method once, performs one-token
   teacher-forced cache decoding, records per-token NLL, and supports resume.
4. Add acceptance and full manifests whose six acceptance cases are reusable
   by the 200-case full matrix.
5. Add CPU tests using fake token streams and fake model outputs. Tests must
   cover corpus drift, anchor math, cache-dependent score accounting, schema
   consistency, expected-ID aggregation, method-load reuse, and dirty-source
   preflight rejection.
6. Document server preflight, acceptance, repeat consistency, full execution,
   validation, and interpretation boundaries in `docs/cage_experiments.md`.
7. Run the full CPU suite locally. On the server, run only acceptance until
   real FP16 incremental-versus-one-shot deltas are observed and reviewed.
