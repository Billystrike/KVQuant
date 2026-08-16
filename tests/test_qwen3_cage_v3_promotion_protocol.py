import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_promotion_protocol import (
    CageV3PromotionProtocolError,
    METHOD_IDS,
    load_promotion_protocol,
    validate_promotion_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_protocol_v1.json"


class CageV3PromotionProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_promotion_protocol(PROTOCOL_PATH)

    def test_freezes_single_candidate_shared_holdout_and_exact_case_counts(self):
        self.assertEqual(len(self.protocol_sha256), 64)
        self.assertEqual([row["method_id"] for row in self.protocol["method_grid"]], list(METHOD_IDS))
        self.assertEqual(self.protocol["input_receipt"]["partition"], "holdout")
        self.assertEqual(self.protocol["input_receipt"]["document_count"], 20)
        self.assertEqual(self.protocol["input_receipt"]["anchor_count"], 40)
        self.assertEqual(
            self.protocol["execution_stages"]["full_holdout"],
            {
                "authorized_only_after_joint_acceptance_gate": True,
                "cage_partition_case_count": 480,
                "kitty_partition_case_count": 120,
                "total_case_count": 600,
            },
        )

    def test_candidate_has_frozen_pareto_memory_signal_without_exceeding_kitty(self):
        methods = {row["method_id"]: row for row in self.protocol["method_grid"]}
        candidate = methods["cage-v3-sr2-sink32-calibrated"]
        predecessor = methods["cage-v1-kittypro-matched"]
        kitty = methods["kitty-pro-25pct"]
        candidate_bytes = [row["packed_bytes"] for row in candidate["points"]]
        predecessor_bytes = [row["packed_bytes"] for row in predecessor["points"]]
        kitty_bytes = [row["packed_bytes"] for row in kitty["points"]]
        self.assertLessEqual(sum(candidate_bytes) / sum(predecessor_bytes) - 1.0, -0.02)
        self.assertTrue(all(c / p - 1.0 <= 0.001 for c, p in zip(candidate_bytes, predecessor_bytes, strict=True)))
        self.assertTrue(all(c <= k for c, k in zip(candidate_bytes, kitty_bytes, strict=True)))

    def test_promotion_requires_predecessor_kivi_and_kitty_gates(self):
        gate = self.protocol["promotion_gate"]
        self.assertTrue(gate["all_three_comparison_gates_must_pass"])
        self.assertEqual(gate["per_length_noninferiority_relative_ppl_percent_at_most"], 0.5)
        self.assertEqual(gate["overall_material_superiority_relative_ppl_percent_at_most"], -1.0)
        self.assertEqual(gate["overall_noninferiority_relative_ppl_percent_at_most"], 0.5)
        self.assertEqual(
            gate["versus_predecessor"]["pass_rule"],
            "quality_superiority_track OR memory_quality_pareto_track",
        )
        self.assertIn("uniform_baseline", gate)
        self.assertIn("external_baseline", gate)

    def test_protocol_blocks_test_llama_and_post_holdout_tuning(self):
        boundary = self.protocol["decision_boundary"]
        self.assertFalse(boundary["pg19_test_access_authorized"])
        self.assertFalse(boundary["llama2_v3_implementation_authorized_before_promotion_pass"])
        self.assertFalse(boundary["kitty_llama_port_authorized_before_promotion_pass"])
        self.assertFalse(boundary["additional_v3_tuning_after_holdout"])
        self.assertFalse(self.protocol["scoring"]["local_perturbation_used_for_selection"])

    def test_static_preflight_cannot_authorize_or_read_gpu_metrics(self):
        source = (
            REPO_ROOT / "scripts" / "qwen3_validate_cage_v3_promotion_protocol.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"reads_holdout_method_metrics": False', source)
        self.assertIn('"gpu_acceptance_authorized_by_this_preflight": False', source)
        self.assertIn('"full_holdout_authorized": False', source)
        self.assertNotIn("case_files", source)
        self.assertNotIn("mean_nll", source)

    def test_rejects_posthoc_threshold_candidate_or_boundary_mutation(self):
        mutations = []
        changed = copy.deepcopy(self.protocol)
        changed["promotion_gate"]["overall_material_superiority_relative_ppl_percent_at_most"] = -0.1
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["method_grid"][3]["method_id"] = "another-candidate"
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["decision_boundary"]["pg19_test_access_authorized"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["execution_stages"]["full_holdout"]["authorized_only_after_joint_acceptance_gate"] = False
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(CageV3PromotionProtocolError):
                    validate_promotion_protocol(changed)

    def test_protocol_file_round_trip_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(CageV3PromotionProtocolError):
                load_promotion_protocol(path)


if __name__ == "__main__":
    unittest.main()
