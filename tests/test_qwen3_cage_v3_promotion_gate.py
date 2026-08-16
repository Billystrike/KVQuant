import copy
import json
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_full import expand_full_holdout_cases
from utils.qwen3_cage_v3_promotion_gate import (
    CageV3PromotionGateError,
    load_promotion_gate,
)
from utils.qwen3_cage_v3_promotion_protocol import load_promotion_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_protocol_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_acceptance_gate_v1.json"


def _holdout_manifest():
    anchors = []
    cases = []
    for document_index in range(20):
        for anchor_index in range(2):
            compact = []
            full_window = list(range(4096))
            for prompt_length in (1024, 2048, 4032):
                case_id = f"holdout-{document_index}-{anchor_index}-{prompt_length}"
                identity = {
                    "partition": "holdout",
                    "document_id": f"document-{document_index:02d}",
                    "anchor_index": anchor_index,
                    "prompt_length": prompt_length,
                }
                cases.append({"case_id": case_id, "identity": identity})
                compact.append({"case_id": case_id, "prompt_length": prompt_length})
            anchors.append(
                {
                    "partition": "holdout",
                    "document_id": f"document-{document_index:02d}",
                    "anchor_index": anchor_index,
                    "full_window_ids": full_window,
                    "cases": compact,
                }
            )
    return {"anchors": anchors, "cases": cases}


class CageV3PromotionGateTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_promotion_protocol(PROTOCOL_PATH)

    def test_checked_in_gate_authorizes_only_frozen_development_holdout_execution(self):
        gate, gate_sha256 = load_promotion_gate(
            GATE_PATH,
            repo_root=REPO_ROOT,
            protocol_sha256=self.protocol_sha256,
            input_manifest_sha256=self.protocol["input_receipt"]["input_manifest_sha256"],
            verify_server_artifacts=False,
        )
        self.assertEqual(len(gate_sha256), 64)
        authorization = gate["full_holdout_authorization"]
        self.assertTrue(authorization["gpu_full_holdout_execution"])
        self.assertFalse(authorization["holdout_interpretation_before_complete_postrun"])
        self.assertFalse(authorization["pg19_test_access"])
        self.assertFalse(authorization["llama2_execution"])
        self.assertFalse(authorization["paper_claims"])

    def test_gate_rejects_receipt_or_scope_mutation(self):
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(gate)
        changed["joint_postrun"]["output_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed["partitions"]["kitty_qwen3"]["full_holdout_case_count"] = 121
        mutations.append(changed)
        changed = copy.deepcopy(gate)
        changed["full_holdout_authorization"]["pg19_test_access"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "gate.json"
                    path.write_text(json.dumps(mutation), encoding="utf-8")
                    with self.assertRaises(CageV3PromotionGateError):
                        load_promotion_gate(
                            path,
                            repo_root=REPO_ROOT,
                            protocol_sha256=self.protocol_sha256,
                            input_manifest_sha256=self.protocol["input_receipt"]["input_manifest_sha256"],
                            verify_server_artifacts=False,
                        )

    def test_full_holdout_expansion_is_exact_and_method_complete(self):
        gate, gate_sha256 = load_promotion_gate(
            GATE_PATH,
            repo_root=REPO_ROOT,
            protocol_sha256=self.protocol_sha256,
            input_manifest_sha256=self.protocol["input_receipt"]["input_manifest_sha256"],
            verify_server_artifacts=False,
        )
        manifest = _holdout_manifest()
        cage = expand_full_holdout_cases(
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
            gate_sha256=gate_sha256,
            manifest=manifest,
            partition="cage_qwen3",
            repo_root=REPO_ROOT,
        )
        kitty = expand_full_holdout_cases(
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
            gate_sha256=gate_sha256,
            manifest=manifest,
            partition="kitty_qwen3",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(len(cage), 480)
        self.assertEqual(len(kitty), 120)
        self.assertEqual({case["input"]["partition"] for case in [*cage, *kitty]}, {"holdout"})
        self.assertEqual(
            {case["method"]["metric_method_id"] for case in cage},
            {"fp16", "kivi-kittypro-matched", "cage-v1-kittypro-matched", "cage-v3-sr2-sink32-calibrated"},
        )
        self.assertEqual({case["method"]["metric_method_id"] for case in kitty}, {"kitty-pro-25pct"})
        self.assertEqual(len({case["case_id"] for case in [*cage, *kitty]}), 600)


if __name__ == "__main__":
    unittest.main()
