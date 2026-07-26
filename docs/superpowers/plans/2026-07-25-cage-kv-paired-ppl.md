# CAGE-KV 50-anchor paired PPL implementation plan

1. Extend the frozen PPL manifest parser with a separate 50-anchor selection
   protocol while preserving the existing acceptance and 200-case matrices.
2. Add ten-case acceptance and 1,000-case full manifests in a separate output
   directory.
3. Verify that acceptance case IDs are a subset of full case IDs and that all
   50 maximum-context windows are non-overlapping.
4. Pre-register the two direct CAGE-versus-KIVI comparisons, paired direction,
   anchor clustering, bootstrap seed, resample count, and interpretation
   boundary before observing GPU results.
5. Add strict 1,000-case validation, deterministic paired tables, auditable
   per-anchor deltas, Markdown summary, and PNG/PDF figures.
6. Run the complete CPU suite locally, then run server preflight and ten-case
   acceptance from a clean committed checkout.
7. Require separate-output acceptance consistency before launching the full
   1,000-case matrix.
8. Independently validate and archive raw and analyzed results before using
   them in the paper evidence table.
9. Post-run amendment (2026-07-26): preserve the frozen matrix after the
   maximum-token FP16 calibration gate failed narrowly, record the deviation
   explicitly, restrict primary analysis to the two declared incremental
   CAGE-versus-KIVI comparisons, and perform the omitted separate-output
   acceptance repeat as a post-full reproducibility audit.
