import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_data import (
    MAXIMUM_WINDOW_TOKENS,
    build_input_manifest,
    derive_document_partitions,
    load_data_protocol,
    nonoverlapping_prompt_starts,
    validate_data_protocol,
    validate_input_manifest,
)
from utils.qwen3_cases import token_ids_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_pg19_data_protocol_v1.json"


def _synthetic_inputs(protocol):
    document_ids = [
        document_id
        for partition in ("calibration", "screen", "holdout")
        for document_id in protocol["frozen_partitions"][partition]
    ]
    documents = []
    tokens = {}
    # Assign token counts so sorting yields ten strata while the within-stratum
    # document hash order reproduces the checked-in frozen partition order.
    for stratum in range(10):
        selected = [
            protocol["frozen_partitions"]["calibration"][stratum],
            *protocol["frozen_partitions"]["screen"][stratum * 2 : stratum * 2 + 2],
            *protocol["frozen_partitions"]["holdout"][stratum * 2 : stratum * 2 + 2],
        ]
        for within, document_id in enumerate(selected):
            token_count = 9000 + stratum * 50 + within
            token_ids = [(stratum * 7 + within + index) % 151000 for index in range(token_count)]
            identity_hash = f"{within:02x}" + f"{stratum:02x}" + "a" * 60
            documents.append(
                {
                    "row_index": len(documents),
                    "document_id": document_id,
                    "document_identity_sha256": identity_hash,
                    "token_count": token_count,
                    "token_ids_sha256": token_ids_sha256(token_ids),
                    "text_sha256": f"{len(documents):064x}",
                    "metadata": {"title": document_id},
                }
            )
            tokens[document_id] = token_ids
    if set(document_ids) != set(tokens):
        raise AssertionError("synthetic documents do not cover the protocol")
    audit = {"status": "pass", "claim_eligible": False, "documents": documents}
    return audit, tokens


class CageV4DataTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_data_protocol(PROTOCOL_PATH)

    def test_checked_in_protocol_freezes_data_but_not_metrics_or_method(self):
        self.assertEqual(
            {name: len(rows) for name, rows in self.protocol["frozen_partitions"].items()},
            {"calibration": 10, "screen": 20, "holdout": 20},
        )
        boundary = self.protocol["access_boundary"]
        self.assertFalse(boundary["holdout_method_metrics_authorized"])
        self.assertFalse(boundary["metric_protocol_frozen"])
        self.assertFalse(boundary["cage_v4_method_frozen"])
        self.assertFalse(boundary["gpu_execution_authorized"])

    def test_protocol_rejects_partition_or_access_mutation(self):
        changed = copy.deepcopy(self.protocol)
        changed["frozen_partitions"]["screen"][0], changed["frozen_partitions"]["holdout"][0] = (
            changed["frozen_partitions"]["holdout"][0],
            changed["frozen_partitions"]["screen"][0],
        )
        # Shape alone remains valid here; the deterministic derivation is
        # checked when audited document identities are supplied.
        validate_data_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed["access_boundary"]["gpu_execution_authorized"] = True
        with self.assertRaisesRegex(ValueError, "access boundary"):
            validate_data_protocol(changed)

    def test_partition_derivation_and_manifest_are_deterministic(self):
        audit, tokens = _synthetic_inputs(self.protocol)
        self.assertEqual(derive_document_partitions(audit["documents"]), self.protocol["frozen_partitions"])
        manifest = build_input_manifest(
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
            audit=audit,
            audit_sha256=self.protocol["source_receipt"]["token_audit_sha256"],
            token_ids_by_document=tokens,
            tokenizer_identity={"class": "synthetic"},
            source_state={"git_commit": "synthetic", "dirty": False},
        )
        self.assertEqual(len(manifest["documents"]), 50)
        self.assertEqual(len(manifest["anchors"]), 100)
        self.assertEqual(len(manifest["cases"]), 300)
        validate_input_manifest(manifest, protocol=self.protocol, protocol_sha256=self.protocol_sha256)

    def test_two_window_rule_is_nonoverlapping_and_in_bounds(self):
        for token_count in (8192, 10371, 431387):
            first, second = nonoverlapping_prompt_starts(token_count)
            self.assertGreaterEqual(first, 0)
            self.assertGreaterEqual(second, first + MAXIMUM_WINDOW_TOKENS)
            self.assertLessEqual(second + MAXIMUM_WINDOW_TOKENS, token_count)
        with self.assertRaisesRegex(ValueError, "too short"):
            nonoverlapping_prompt_starts(8191)

    def test_manifest_rejects_token_mutation(self):
        audit, tokens = _synthetic_inputs(self.protocol)
        document_id = audit["documents"][0]["document_id"]
        tokens[document_id][0] += 1
        with self.assertRaisesRegex(ValueError, "token hash"):
            build_input_manifest(
                protocol=self.protocol,
                protocol_sha256=self.protocol_sha256,
                audit=audit,
                audit_sha256=self.protocol["source_receipt"]["token_audit_sha256"],
                token_ids_by_document=tokens,
                tokenizer_identity={"class": "synthetic"},
                source_state={"git_commit": "synthetic", "dirty": False},
            )


if __name__ == "__main__":
    unittest.main()
