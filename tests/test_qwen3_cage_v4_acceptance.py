import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.qwen3_compare_cage_v4_metric_acceptance import _payload
from utils.qwen3_cage_v4_acceptance import (
    acceptance_input_cases,
    expand_metric_methods,
)
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_validity_protocol_v1.json"


def _minimal_manifest(protocol):
    document_id = protocol["acceptance_policy"]["document_id"]
    full_window = list(range(4096))
    cases = []
    compact = []
    for length in (1024, 2048, 4032):
        case_id = f"case-{length}"
        identity = {
            "partition": "screen",
            "document_id": document_id,
            "anchor_index": 0,
            "prompt_length": length,
        }
        cases.append({"case_id": case_id, "identity": identity})
        compact.append({"case_id": case_id, "prompt_length": length})
    return {
        "anchors": [
            {
                "document_id": document_id,
                "anchor_index": 0,
                "partition": "screen",
                "full_window_ids": full_window,
                "cases": compact,
            }
        ],
        "cases": cases,
    }


class CageV4AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_metric_protocol(PROTOCOL_PATH)

    def test_method_expansion_matches_frozen_partition_counts_and_bytes(self):
        cage = expand_metric_methods(self.protocol, repo_root=REPO_ROOT, partition="cage_qwen3")
        kitty = expand_metric_methods(self.protocol, repo_root=REPO_ROOT, partition="kitty_qwen3")
        self.assertEqual(len(cage), 15)
        self.assertEqual(len(kitty), 3)
        self.assertEqual({row["metric_method_id"] for row in kitty}, {"kitty-pro-25pct"})
        frozen = {
            (method["method_id"], point["prompt_length"]): point["packed_bytes"]
            for method in self.protocol["method_grid"]
            for point in method["points"]
        }
        for method in [*cage, *kitty]:
            self.assertEqual(
                method["packed_bytes"],
                frozen[(method["metric_method_id"], method["prompt_length"])],
            )

    def test_acceptance_input_is_one_document_anchor_with_nested_prompts(self):
        manifest = _minimal_manifest(self.protocol)
        inputs = acceptance_input_cases(manifest, protocol=self.protocol)
        self.assertEqual(set(inputs), {1024, 2048, 4032})
        self.assertEqual(len(inputs[1024]["prompt_ids"]), 1024)
        self.assertEqual(len(inputs[2048]["prompt_ids"]), 2048)
        self.assertEqual(len(inputs[4032]["prompt_ids"]), 4032)
        self.assertEqual(inputs[1024]["continuation_ids"], list(range(4032, 4096)))
        self.assertEqual(inputs[1024]["continuation_ids"], inputs[4032]["continuation_ids"])

    def test_acceptance_input_rejects_missing_or_duplicate_anchor(self):
        manifest = _minimal_manifest(self.protocol)
        manifest["anchors"] = []
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            acceptance_input_cases(manifest, protocol=self.protocol)
        manifest = _minimal_manifest(self.protocol)
        manifest["anchors"].append(copy.deepcopy(manifest["anchors"][0]))
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            acceptance_input_cases(manifest, protocol=self.protocol)

    def test_repeat_payload_excludes_runtime_and_provenance_fields(self):
        scientific = {
            "case_id": "case-a",
            "partition": "cage_qwen3",
            "method": {"id": "fp16-l1024"},
            "input": {"prompt_length": 1024},
            "memory": {"model_total_bytes": 1},
            "scoring": {"mean_nll": 2.0},
            "local_perturbation": None,
            "quality_cache": {"reported_seq_length": 1087},
        }
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("a", "b")]
            for index, root in enumerate(roots):
                (root / "cases").mkdir(parents=True)
                record = {
                    **scientific,
                    "identity": {"repeat": index},
                    "model": {"loaded_at": index},
                    "runtime": {"elapsed_seconds": index + 1.0},
                    "completed_at_utc": f"2026-08-10T00:00:0{index}+00:00",
                }
                (root / "cases" / "case-a.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
            self.assertEqual(_payload(roots[0]), _payload(roots[1]))

    def test_gpu_runner_exposes_acceptance_only(self):
        source = (
            REPO_ROOT / "scripts" / "qwen3_run_cage_v4_metric_acceptance.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("holdout", source.lower())


if __name__ == "__main__":
    unittest.main()
