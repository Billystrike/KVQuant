import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v3_audit import candidate_selection_audit, token_ids_sha256
from utils.qwen3_cage_v3_manifest import build_cage_v3_manifest, validate_cage_v3_manifest
from utils.qwen3_cage_v3_protocol import (
    TOTAL_TWO_BIT_CHANNELS,
    calibrated_tiered_two_bit_quotas,
    file_sha256,
    load_cage_v3_protocol,
    validate_cage_v3_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_development_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "c92e452b5eac99c7a015da21a82080cf61ddd070e677cc3664ed78bf657cbfe9"


class CageV3ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_cage_v3_protocol(PROTOCOL_PATH)

    def test_checked_in_protocol_hash_candidates_and_boundaries_are_frozen(self):
        self.assertEqual(file_sha256(PROTOCOL_PATH), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.protocol_sha256, EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(len(self.protocol["candidate_families"]), 3)
        self.assertEqual(self.protocol["screen_gate"]["maximum_candidates_advanced"], 1)
        boundary = self.protocol["development_data"]["partition_boundary"]
        self.assertFalse(boundary["reserved_unseen_execution_authorized"])
        self.assertFalse(self.protocol["fixed_quantization"]["value_adaptive"])
        self.assertFalse(self.protocol["method_boundary"]["one_bit_refinement_enabled"])

    def test_calibrated_quota_is_stable_tiered_and_byte_preserving(self):
        scores = [1.0] * 36
        quotas = calibrated_tiered_two_bit_quotas(scores)
        self.assertEqual(quotas[:12], [48] * 12)
        self.assertEqual(quotas[12:24], [32] * 12)
        self.assertEqual(quotas[24:], [16] * 12)
        self.assertEqual(sum(quotas), TOTAL_TWO_BIT_CHANNELS)
        ranked = calibrated_tiered_two_bit_quotas(list(reversed(range(36))))
        self.assertEqual(ranked[0], 48)
        self.assertEqual(ranked[35], 16)
        with self.assertRaisesRegex(ValueError, "36"):
            calibrated_tiered_two_bit_quotas([1.0] * 35)

    def test_protocol_rejects_posthoc_partition_or_method_mutation(self):
        changed = copy.deepcopy(self.protocol)
        changed["development_data"]["partitions"]["holdout"] = list(range(9, 19))
        with self.assertRaisesRegex(ValueError, "partitions"):
            validate_cage_v3_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed["method_boundary"]["one_bit_refinement_enabled"] = True
        with self.assertRaisesRegex(ValueError, "one-bit"):
            validate_cage_v3_protocol(changed)

    def test_synthetic_manifest_binds_all_50_anchors_and_150_cases(self):
        protocol = copy.deepcopy(self.protocol)
        token_ids = list(range(300000))
        data = protocol["development_data"]
        data["token_count"] = len(token_ids)
        data["token_ids_sha256"] = token_ids_sha256(token_ids)
        data["snapshot_sha256"] = "a" * 64
        selection = candidate_selection_audit(token_ids, 50)
        data["minimum_anchor_gap"] = selection["minimum_anchor_gap"]
        audit = {
            "status": "pass",
            "claim_eligible": False,
            "corpus": {"token_ids_sha256": data["token_ids_sha256"]},
            "candidate_selection_audits": [selection],
        }
        manifest = build_cage_v3_manifest(
            protocol=protocol,
            protocol_sha256="b" * 64,
            token_ids=token_ids,
            snapshot_sha256="a" * 64,
            token_audit=audit,
            tokenizer_identity={"class": "synthetic"},
            source_state={"git_commit": "synthetic", "dirty": False},
        )
        self.assertEqual(len(manifest["selection"]["anchors"]), 50)
        self.assertEqual(len(manifest["cases"]), 150)
        self.assertEqual(manifest["selection"]["partitions"], data["partitions"])
        validate_cage_v3_manifest(
            manifest,
            protocol=protocol,
            protocol_sha256="b" * 64,
        )


if __name__ == "__main__":
    unittest.main()
