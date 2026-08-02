import json
import tempfile
import unittest
from pathlib import Path

import scripts.cage_run_passkey as passkey_runner
from tests.test_cage_passkey import (
    FakeTokenizer,
    completed_record,
    source_state,
)
from utils.cage_experiment_io import atomic_write_json
from utils.cage_passkey import (
    aggregate_passkey_cases,
    expand_passkey_cases,
    load_passkey_manifest,
    resolved_passkey_manifest,
)
from utils.cage_passkey_analysis import (
    PasskeyAnalysisError,
    aggregate_passkey_results,
    load_completed_passkey_matrix,
    wilson_interval,
    write_passkey_analysis_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def build_frozen_results(root: Path):
    manifest = load_passkey_manifest(
        ROOT / "configs" / "cage_passkey_llama2_7b_stage_b.json"
    )
    keys, cases = expand_passkey_cases(manifest, FakeTokenizer(), source_state())
    resolved = resolved_passkey_manifest(
        manifest,
        generated_keys=keys,
        cases=cases,
        source_state=source_state(),
    )
    atomic_write_json(root / "manifest.resolved.json", resolved)
    (root / "cases").mkdir(parents=True)
    (root / "failures").mkdir()

    miss_methods = {"kivi-g32-r32", "cage-r32", "cage-r64"}
    records = []
    for case in cases:
        is_shared_miss = (
            case["method"]["id"] in miss_methods
            and case["input"]["prompt_length"] == 2048
            and case["input"]["position_percent"] == 50
            and case["input"]["key_index"] == 2
        )
        response = "338966." if is_shared_miss else None
        record = completed_record(
            case,
            response_text=response,
            model_reference=manifest["model"]["reference"],
        )
        self_path = root / "cases" / f"{record['case_id']}.json"
        self_path.write_text(json.dumps(record), encoding="utf-8")
        records.append(record)

    completed = aggregate_passkey_cases(
        root,
        expected_case_ids=[case["case_id"] for case in cases],
    )
    result = {
        "protocol_stage": "stage_b",
        "expanded_cases": 300,
        "completed_cases": 300,
        "failure_records": 0,
    }
    passkey_runner._attach_quality_counts(result, completed)
    passkey_runner._attach_quality_summary(result, manifest, completed)
    passkey_runner._write_quality_summary(root, result)
    return resolved, records


class PasskeyAnalysisTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_non_degenerate_at_perfect_accuracy(self):
        low, high = wilson_interval(60, 60)
        self.assertGreater(low, 0.9)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_loads_aggregates_and_writes_declared_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            resolved, _ = build_frozen_results(root)
            loaded_resolved, records, quality = load_completed_passkey_matrix(root)
            self.assertEqual(loaded_resolved, resolved)
            self.assertEqual(len(records), 300)
            self.assertEqual(quality["exact_matches"], 297)

            tables = aggregate_passkey_results(records, loaded_resolved)
            self.assertEqual(len(tables["method_summary"]), 5)
            self.assertEqual(len(tables["length_summary"]), 20)
            self.assertEqual(len(tables["position_summary"]), 15)
            self.assertEqual(len(tables["cell_summary"]), 60)
            self.assertEqual(len(tables["paired_comparisons"]), 10)
            self.assertEqual(len(tables["exact_misses"]), 3)

            by_method = {
                row["method_id"]: row for row in tables["method_summary"]
            }
            self.assertEqual(by_method["fp16"]["exact_matches"], 60)
            self.assertEqual(by_method["kivi-g32-r32"]["exact_matches"], 59)
            pair = next(
                row
                for row in tables["paired_comparisons"]
                if row["method_a_id"] == "fp16"
                and row["method_b_id"] == "cage-r32"
            )
            self.assertEqual(pair["method_a_only_exact"], 1)
            self.assertEqual(pair["method_b_only_exact"], 0)

            output = Path(directory) / "analysis"
            paths = write_passkey_analysis_outputs(
                output,
                tables,
                resolved_manifest=loaded_resolved,
                quality_summary=quality,
                make_plots=False,
            )
            self.assertEqual(len(paths), 14)
            self.assertTrue((output / "method_summary.csv").is_file())
            self.assertTrue((output / "paired_comparisons.jsonl").is_file())
            self.assertTrue((output / "exact_misses.csv").is_file())
            protocol = json.loads((output / "analysis_protocol.json").read_text())
            self.assertEqual(protocol["case_count"], 300)
            self.assertIn(
                "five deterministic keys",
                protocol["sampling_boundary"],
            )
            markdown = (output / "passkey_summary.md").read_text()
            self.assertIn("kivi-g64-r64", markdown)
            self.assertIn("338966", markdown)

    def test_rejects_nonempty_analysis_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(PasskeyAnalysisError, "not empty"):
                write_passkey_analysis_outputs(
                    output,
                    {
                        "method_summary": [],
                        "length_summary": [],
                        "position_summary": [],
                        "cell_summary": [],
                        "paired_comparisons": [],
                        "exact_misses": [],
                    },
                    resolved_manifest={},
                    quality_summary={},
                    make_plots=False,
                )


if __name__ == "__main__":
    unittest.main()
