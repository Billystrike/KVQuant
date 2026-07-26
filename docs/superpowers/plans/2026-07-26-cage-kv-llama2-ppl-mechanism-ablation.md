# Llama-2-7B paired PPL mechanism ablation execution plan

1. Add strict manifest identities for the 20-case acceptance and 2,000-case
   full mechanism-ablation matrices.
2. Run the complete CPU suite and a server preflight that proves acceptance is
   a strict subset of full and that all ten configurations match the frozen
   local ablation.
3. Execute acceptance, validate 20 completed cases and zero failures, then
   rerun it to verify resume behavior.
4. Execute the full 2,000-case matrix and validate 126,000 cache-dependent
   targets, method/length/anchor coverage, finite metrics, and zero failures.
5. Back up raw JSON/JSONL/CSV, resolved manifest, logs, source, corpus identity,
   and environment information before analysis.
6. Generate the pre-registered paired mechanism tables, descriptive bootstrap
   intervals, figures, and a claim-bounded Markdown summary.
7. Integrate the local perturbation and paired PPL mechanism evidence into the
   Llama-2-7B paper evidence ledger without replacing the frozen direct
   CAGE-versus-KIVI results.
