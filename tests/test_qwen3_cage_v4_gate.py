import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_data import file_sha256
from utils.qwen3_cage_v4_gate import CageV4GateError, load_acceptance_gate
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol
from utils.qwen3_cage_v4_screen import expand_full_screen_cases


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_validity_protocol_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_acceptance_gate_v1.json"


def _screen_manifest():
    anchors = []
    cases = []
    for document_index in range(20):
        for anchor_index in range(2):
            compact = []
            full_window = list(range(4096))
            for prompt_length in (1024, 2048, 4032):
                case_id = f"screen-{document_index}-{anchor_index}-{prompt_length}"
                identity = {
                    "partition": "screen",
                    "document_id": f"document-{document_index}",
                    "anchor_index": anchor_index,
                    "prompt_length": prompt_length,
                }
                cases.append({"case_id": case_id, "identity": identity})
                compact.append({"case_id": case_id, "prompt_length": prompt_length})
            anchors.append(
                {
                    "partition": "screen",
                    "document_id": f"document-{document_index}",
                    "anchor_index": anchor_index,
                    "full_window_ids": full_window,
                    "cases": compact,
                }
            )
    return {"anchors": anchors, "cases": cases}


class CageV4GateTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_metric_protocol(PROTOCOL_PATH)

    def test_checked_in_gate_freezes_both_repeat_pairs_and_boundaries(self):
        gate, gate_sha256 = load_acceptance_gate(
            GATE_PATH,
            repo_root=REPO_ROOT,
            protocol_sha256=self.protocol_sha256,
            input_manifest_sha256=self.protocol["input_receipt"]["input_manifest_sha256"],
            verify_artifacts=False,
        )
        self.assertEqual(gate_sha256, file_sha256(GATE_PATH))
        self.assertEqual(gate["partitions"]["cage_qwen3"]["case_count"], 15)
        self.assertEqual(gate["partitions"]["kitty_qwen3"]["case_count"], 3)
        authorization = gate["full_screen_authorization"]
        self.assertEqual(authorization["input_partition"], "screen")
        self.assertFalse(authorization["pg19_holdout_method_metrics"])
        self.assertFalse(authorization["pg19_test_access"])
        self.assertFalse(authorization["cage_v4_candidate_execution"])

    def test_gate_rejects_posthoc_authorization_or_receipt_mutation(self):
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(gate)
        changed["full_screen_authorization"]["pg19_holdout_method_metrics"] = True
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed["partitions"]["kitty_qwen3"]["scientific_payload_sha256"] = "0" * 64
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "gate.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(CageV4GateError):
                        load_acceptance_gate(
                            path,
                            repo_root=REPO_ROOT,
                            protocol_sha256=self.protocol_sha256,
                            input_manifest_sha256=self.protocol["input_receipt"][
                                "input_manifest_sha256"
                            ],
                            verify_artifacts=False,
                        )

    def test_full_screen_expansion_is_exact_and_contains_no_holdout(self):
        manifest = _screen_manifest()
        cage = expand_full_screen_cases(
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
            manifest=manifest,
            partition="cage_qwen3",
            repo_root=REPO_ROOT,
        )
        kitty = expand_full_screen_cases(
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
            manifest=manifest,
            partition="kitty_qwen3",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(len(cage), 600)
        self.assertEqual(len(kitty), 120)
        self.assertEqual(sum(row["method"]["metric_method_id"] != "fp16" for row in cage), 480)
        self.assertEqual(sum(row["method"]["metric_method_id"] != "fp16" for row in kitty), 120)
        self.assertEqual({row["input"]["partition"] for row in [*cage, *kitty]}, {"screen"})


if __name__ == "__main__":
    unittest.main()
