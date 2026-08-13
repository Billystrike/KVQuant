import copy
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_dtqi_protocol import (
    PACKED_BYTES,
    PROMPT_LENGTHS,
    RESIDUAL_LENGTHS,
    load_dtqi_protocol,
    validate_dtqi_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "8e5478aa75dd2a8685d1b3839b4260583f44949022df1365678d1f0ee476e549"


class CageV4DTQIProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_dtqi_protocol(
            PROTOCOL_PATH,
            repo_root=REPO_ROOT,
        )

    def test_single_candidate_binds_recent_window_to_each_frozen_residual(self):
        self.assertEqual(self.protocol_sha256, EXPECTED_PROTOCOL_SHA256)
        points = self.protocol["candidate"]["points"]
        self.assertEqual(
            [point["prompt_length"] for point in points],
            list(PROMPT_LENGTHS),
        )
        self.assertEqual(
            [point["residual_length"] for point in points],
            list(RESIDUAL_LENGTHS),
        )
        self.assertEqual(
            [point["recent_query_window"] for point in points],
            list(RESIDUAL_LENGTHS),
        )
        self.assertEqual([point["packed_bytes"] for point in points], list(PACKED_BYTES))
        self.assertTrue(self.protocol["candidate"]["only_candidate"])

    def test_storage_success_policy_and_execution_boundary_are_frozen(self):
        storage = self.protocol["candidate"]["inherited_storage"]
        self.assertFalse(storage["packed_representation_changed"])
        self.assertFalse(storage["value_adaptive"])
        self.assertEqual(storage["one_bit_channels"], 0)
        self.assertEqual(
            self.protocol["success_policy"]["overall_material_superiority_relative_ppl_percent"],
            -1.0,
        )
        boundary = self.protocol["execution_boundary"]
        self.assertTrue(boundary["cpu_acceptance_authorized"])
        self.assertFalse(boundary["gpu_acceptance_authorized"])
        self.assertFalse(boundary["gpu_full_screen_authorized"])
        self.assertFalse(boundary["holdout_method_metrics_authorized"])

    def test_protocol_rejects_window_weight_candidate_or_holdout_mutation(self):
        changed = copy.deepcopy(self.protocol)
        changed["candidate"]["points"][0]["recent_query_window"] = 128
        with self.assertRaisesRegex(ValueError, "point definition"):
            validate_dtqi_protocol(changed, repo_root=REPO_ROOT)
        changed = copy.deepcopy(self.protocol)
        changed["candidate"]["key_importance"]["recent_query_weight"] = 0.75
        with self.assertRaisesRegex(ValueError, "importance"):
            validate_dtqi_protocol(changed, repo_root=REPO_ROOT)
        changed = copy.deepcopy(self.protocol)
        changed["candidate"]["only_candidate"] = False
        with self.assertRaisesRegex(ValueError, "only candidate"):
            validate_dtqi_protocol(changed, repo_root=REPO_ROOT)
        changed = copy.deepcopy(self.protocol)
        changed["execution_boundary"]["holdout_method_metrics_authorized"] = True
        with self.assertRaisesRegex(ValueError, "execution boundary"):
            validate_dtqi_protocol(changed, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
