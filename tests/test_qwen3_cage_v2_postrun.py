import copy
import json
import unittest
from pathlib import Path

from utils.qwen3_cage_v2_postrun import (
    CageV2PostrunError,
    validate_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v2_round1_screen_artifacts_v1.json"


class CageV2PostrunTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    def _validate(self, value):
        validate_artifact_manifest(
            value,
            protocol_sha256="5b0aa6f4eade3e42a99bbc731944113ab5f9173489ab78632b5d837eb70218c9",
            input_manifest_sha256="444db800195a515ec33567682fd508b9051920f637d24ffa56f78b19dbaa31ef",
            gate_sha256="5625705019e6a063985d85cf581a57a58027d1ea72886912d11d42e3bb58dd25",
        )

    def test_checked_in_artifacts_freeze_145_cases_and_5220_layers(self):
        self._validate(self.artifacts)
        self.assertEqual(self.artifacts["joint_expected"]["case_count"], 145)
        self.assertEqual(self.artifacts["joint_expected"]["layer_record_count"], 5220)
        self.assertFalse(self.artifacts["claim_eligible"])

    def test_artifact_manifest_rejects_hash_count_and_gate_mutation(self):
        for mutation, message in (
            (("partitions", "cage_qwen3", "case_count", 114), "case count"),
            (("joint_expected", "layer_record_count", 5219), "expectations"),
            (("acceptance_gate_sha256", "0" * 64), "gate hash"),
        ):
            value = copy.deepcopy(self.artifacts)
            if len(mutation) == 4:
                value[mutation[0]][mutation[1]][mutation[2]] = mutation[3]
            elif len(mutation) == 3:
                value[mutation[0]][mutation[1]] = mutation[2]
            else:
                value[mutation[0]] = mutation[1]
            with self.assertRaisesRegex(CageV2PostrunError, message):
                self._validate(value)


if __name__ == "__main__":
    unittest.main()
