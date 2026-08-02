import json
import tempfile
import unittest
from pathlib import Path

from utils.cage_llama2_paper_package import (
    Llama2PaperPackageError,
    build_final_paper_tables,
    load_final_paper_inputs,
    write_final_paper_package,
)


RESIDUALS = (64, 128)
ROLES = ("fixed-random", "uniform", "k-adaptive", "v-adaptive")
LOCAL_LENGTHS = (512, 1024, 2048, 4095)
PPL_LENGTHS = (512, 1024, 2048, 4032)


def _state(commit):
    return {
        "git_commit": commit,
        "dirty": False,
        "dirty_sha256": None,
        "untracked_paths": [],
    }


def _inputs():
    operating = [
        {"comparison_id": comparison, "pareto_prompt_length": length}
        for comparison in (
            "cage-r128_vs_kivi-g32-r128",
            "cage-r64_vs_kivi-g64-r64",
        )
        for length in LOCAL_LENGTHS
    ]
    comparisons = [
        {
            "comparison_id": "cage-r128_vs_kivi-g32-r128",
            "comparison_order": 0,
            "memory_savings_percent_min": 2.0,
            "memory_savings_percent_max": 5.5,
            "overall_candidate_ppl_change_percent_vs_baseline": 0.13,
            "ppl_status": "descriptive_parity",
        },
        {
            "comparison_id": "cage-r64_vs_kivi-g64-r64",
            "comparison_order": 1,
            "memory_savings_percent_min": -11.6,
            "memory_savings_percent_max": -11.1,
            "overall_candidate_ppl_change_percent_vs_baseline": -0.79,
            "ppl_status": "descriptive_improvement",
        },
    ]
    local = []
    role_error = {
        "fixed-random": -30.0,
        "uniform": -25.0,
        "k-adaptive": -5.0,
        "v-adaptive": -20.0,
    }
    role_memory = {
        "fixed-random": 0.0,
        "uniform": 8.0,
        "k-adaptive": 4.0,
        "v-adaptive": 2.0,
    }
    for residual in RESIDUALS:
        for length in LOCAL_LENGTHS:
            for order, role in enumerate(ROLES):
                local.append({
                    "contrast_id": f"full_vs_{role}",
                    "residual_length": residual,
                    "prompt_length": length,
                    "paper_memory_change_vs_baseline_percent": role_memory[role],
                    "candidate_error_change_vs_baseline_percent": role_error[role],
                    "baseline_error_penalty_vs_full_percent": -role_error[role] / 0.75,
                    "full_better_samples": 3,
                    "sample_count": 3,
                    "same_paper_and_runtime_budget": role == "fixed-random",
                    "contrast_order": order,
                })
    local_factorial = []
    for residual in RESIDUALS:
        for length in LOCAL_LENGTHS:
            local_factorial.append({
                "residual_length": residual,
                "prompt_length": length,
                "full_error_reduction_vs_uniform_percent": 25.0,
                "key_only_error_reduction_vs_uniform_percent": 20.0,
                "value_only_error_reduction_vs_uniform_percent": 5.0,
                "error_interaction_full_minus_key_minus_value_plus_uniform": 0.0,
            })
    ppl = []
    role_delta = {
        "fixed-random": -0.20,
        "uniform": -0.03,
        "k-adaptive": -0.001,
        "v-adaptive": -0.029,
    }
    for residual in RESIDUALS:
        for role in ROLES:
            for length in (None, *PPL_LENGTHS):
                delta = role_delta[role]
                low, high = delta - 0.01, delta + 0.01
                if role == "k-adaptive":
                    low, high = -0.004, 0.002
                ppl.append({
                    "comparison_id": f"r{residual}_full_vs_{role}",
                    "comparison_order": ROLES.index(role),
                    "residual_length": residual,
                    "baseline_role": role,
                    "scope": "overall_anchor_clustered" if length is None else "prompt_length",
                    "prompt_length": length,
                    "mean_paired_delta_nll": delta,
                    "candidate_ppl_ratio_to_baseline": pow(2.718281828459045, delta),
                    "bootstrap_mean_delta_nll_ci95_low": low,
                    "bootstrap_mean_delta_nll_ci95_high": high,
                    "candidate_better_anchors": 45 if role != "k-adaptive" else 26,
                    "candidate_worse_anchors": 5 if role != "k-adaptive" else 24,
                })
    ppl_factorial = []
    effect_values = {
        "full_minus_uniform": -0.03,
        "key_adaptive_minus_uniform": -0.029,
        "value_adaptive_minus_uniform": -0.001,
        "interaction_full_minus_key_minus_value_plus_uniform": 0.0,
    }
    for residual in RESIDUALS:
        for length in (None, *PPL_LENGTHS):
            row = {
                "residual_length": residual,
                "scope": "overall_anchor_clustered" if length is None else "prompt_length",
                "prompt_length": length,
            }
            for effect, value in effect_values.items():
                row[f"mean_{effect}"] = value
                row[f"bootstrap_{effect}_ci95_low"] = value - 0.005
                row[f"bootstrap_{effect}_ci95_high"] = value + 0.005
            ppl_factorial.append(row)
    return {
        "evidence_protocol": {
            "schema_version": 1,
            "model_scope": "Llama-2-7B-hf",
            "operating_point_row_count": 8,
            "input_source_states": {"pareto": [_state("pareto")]},
            "paired_ppl_protocol_deviation": {
                "overall_gate": "FAIL",
                "protocol_status": "RECORDED_DEVIATION",
            },
        },
        "operating_point_evidence": operating,
        "comparison_summary": comparisons,
        "local_protocol": {
            "schema_version": 1,
            "input_run_count": 144,
            "paired_contrast_count": 32,
            "factorial_row_count": 8,
            "primary_error_metric": "joint_post_o_proj_mse",
            "source_state": _state("local"),
        },
        "local_contrasts": local,
        "local_factorial": local_factorial,
        "ppl_protocol": {
            "schema_version": 1,
            "protocol_stage": "mechanism_ablation_full",
            "case_count": 2000,
            "anchor_count": 50,
            "bootstrap": {"resamples": 10_000},
            "source_state": _state("ppl"),
        },
        "ppl_comparisons": ppl,
        "ppl_factorial": ppl_factorial,
        "input_directories": {
            "prior_paper_evidence": "/evidence",
            "local_mechanism": "/local",
            "ppl_mechanism": "/ppl",
        },
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_fixture(root, inputs):
    evidence, local, ppl = root / "evidence", root / "local", root / "ppl"
    for path in (evidence, local, ppl):
        path.mkdir()
    _write_json(evidence / "evidence_protocol.json", inputs["evidence_protocol"])
    _write_jsonl(evidence / "operating_point_evidence.jsonl", inputs["operating_point_evidence"])
    _write_jsonl(evidence / "comparison_summary.jsonl", inputs["comparison_summary"])
    _write_json(local / "analysis_protocol.json", inputs["local_protocol"])
    _write_jsonl(local / "paired_contrasts.jsonl", inputs["local_contrasts"])
    _write_jsonl(local / "factorial_decomposition.jsonl", inputs["local_factorial"])
    _write_json(ppl / "analysis_protocol.json", inputs["ppl_protocol"])
    _write_jsonl(ppl / "paired_comparisons.jsonl", inputs["ppl_comparisons"])
    _write_jsonl(ppl / "factorial_summary.jsonl", inputs["ppl_factorial"])
    return evidence, local, ppl


class Llama2PaperPackageTests(unittest.TestCase):
    def test_strict_loader_accepts_the_frozen_package_shapes(self):
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as directory:
            evidence, local, ppl = _write_fixture(Path(directory), inputs)
            loaded = load_final_paper_inputs(evidence, local, ppl)
            tables = build_final_paper_tables(loaded)
            self.assertEqual(len(tables["mechanism_length_evidence"]), 40)
            self.assertEqual(len(tables["claim_register"]), 9)

    def test_builds_final_tables_and_preserves_native_context_mismatch(self):
        tables = build_final_paper_tables(_inputs())
        self.assertEqual(len(tables["operating_point_evidence"]), 8)
        self.assertEqual(len(tables["comparison_summary"]), 2)
        self.assertEqual(len(tables["mechanism_length_evidence"]), 40)
        self.assertEqual(len(tables["mechanism_summary"]), 8)
        self.assertEqual(len(tables["factorial_evidence"]), 2)
        self.assertEqual(len(tables["claim_register"]), 9)
        statuses = [row["cross_metric_join_status"] for row in tables["mechanism_length_evidence"]]
        self.assertEqual(statuses.count("exact_prompt_length"), 24)
        self.assertEqual(statuses.count("local_only_4095"), 8)
        self.assertEqual(statuses.count("ppl_only_4032"), 8)

    def test_classifies_key_dominance_without_equivalence_claim(self):
        tables = build_final_paper_tables(_inputs())
        summary = {
            (row["residual_length"], row["baseline_role"]): row
            for row in tables["mechanism_summary"]
        }
        self.assertEqual(
            summary[(64, "fixed-random")]["claim_classification"],
            "same_budget_importance_ordering_supported",
        )
        self.assertEqual(
            summary[(64, "k-adaptive")]["claim_classification"],
            "full_vs_key_only_descriptive_parity",
        )
        key_claim = next(
            row for row in tables["claim_register"]
            if row["claim_id"] == "key_dominant_mechanism"
        )
        self.assertIn("Do not claim", key_claim["prohibited_extension"])
        self.assertNotIn("equivalent", key_claim["claim"].lower())

    def test_writes_sixteen_nonplot_outputs_and_rejects_nonempty_target(self):
        inputs = _inputs()
        tables = build_final_paper_tables(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package"
            outputs = write_final_paper_package(
                output, inputs, tables, make_plots=False,
            )
            self.assertEqual(len(outputs), 16)
            self.assertEqual(
                len((output / "mechanism_length_evidence.jsonl").read_text().splitlines()),
                40,
            )
            self.assertIn("Key adaptation", (output / "paper_results_draft.md").read_text())
            self.assertIn("\\begin{tabular}", (output / "paper_tables.tex").read_text())
            with self.assertRaisesRegex(Llama2PaperPackageError, "not empty"):
                write_final_paper_package(output, inputs, tables, make_plots=False)

    def test_strict_loader_rejects_dirty_input_source(self):
        inputs = _inputs()
        inputs["ppl_protocol"]["source_state"]["dirty"] = True
        with tempfile.TemporaryDirectory() as directory:
            evidence, local, ppl = _write_fixture(Path(directory), inputs)
            with self.assertRaisesRegex(Llama2PaperPackageError, "clean source state"):
                load_final_paper_inputs(evidence, local, ppl)


if __name__ == "__main__":
    unittest.main()
