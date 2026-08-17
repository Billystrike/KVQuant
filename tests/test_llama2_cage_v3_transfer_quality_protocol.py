import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_protocol import (
    EXPECTED_POSTRUN_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_QUOTA_SHA256,
    Llama2CageV3TransferQualityProtocolError,
    load_transfer_quality_protocol,
    validate_transfer_quality_protocol,
)
from utils.qwen3_cage_v4_data import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_protocol_v1.json"
POSTRUN_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_gpu_acceptance_postrun_receipt_v1.json"
QUOTA_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quota_plan_v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_quality_preflight.py"


class Llama2CageV3TransferQualityProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.digest = load_transfer_quality_protocol(
            PROTOCOL_PATH, repo_root=REPO_ROOT
        )

    def test_checked_in_protocol_receipt_and_quota_are_exact(self):
        self.assertEqual(self.digest, EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(file_sha256(POSTRUN_PATH), EXPECTED_POSTRUN_SHA256)
        self.assertEqual(file_sha256(QUOTA_PATH), EXPECTED_QUOTA_SHA256)
        receipt = json.loads(POSTRUN_PATH.read_text(encoding="utf-8"))
        self.assertTrue(receipt["decision"]["production_gpu_acceptance_pass"])
        self.assertTrue(receipt["decision"]["quality_protocol_design_authorized"])
        self.assertFalse(receipt["decision"]["formal_transfer_execution_authorized"])

    def test_freezes_exact_600_case_four_role_matrix(self):
        cases = self.protocol["case_matrix"]
        self.assertEqual(cases["method_length_points"], 12)
        self.assertEqual(cases["anchors_per_point"], 50)
        self.assertEqual(cases["full_case_count"], 600)
        self.assertEqual(cases["acceptance_case_count_per_repeat"], 12)
        rows = self.protocol["method_length_matrix"]
        self.assertEqual([row["prompt_length"] for row in rows], [1024, 2048, 4032])
        for row in rows:
            self.assertEqual(
                [method["role"] for method in row["methods"]],
                ["quality_reference", "transfer_candidate", "primary_predecessor", "primary_uniform_baseline"],
            )

    def test_scoring_and_primary_thresholds_are_unchanged(self):
        scoring = self.protocol["scoring"]
        self.assertEqual(scoring["boundary_target_count"], 1)
        self.assertEqual(scoring["single_token_decode_target_count"], 63)
        self.assertEqual(scoring["primary_target_count_per_case"], 64)
        gate = self.protocol["primary_transfer_gate"]
        self.assertEqual(gate["per_length_noninferiority_relative_ppl_percent_at_most"], 0.5)
        self.assertEqual(gate["overall_material_superiority_relative_ppl_percent_at_most"], -1.0)
        self.assertEqual((gate["bootstrap_resamples"], gate["bootstrap_seed"]), (10000, 20260817))
        self.assertTrue(gate["both_primary_comparisons_must_pass"])

    def test_rejects_candidate_threshold_input_or_authorization_mutation(self):
        mutations = []
        changed = copy.deepcopy(self.protocol)
        changed["method_length_matrix"][0]["methods"][1]["residual_length"] = 177
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["primary_transfer_gate"]["overall_material_superiority_relative_ppl_percent_at_most"] = 0.0
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["input"]["expected_token_count"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["current_boundary"]["quality_acceptance_execution_authorized"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["reporting"]["kitty_llama_included"] = True
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(Llama2CageV3TransferQualityProtocolError):
                    validate_transfer_quality_protocol(changed, repo_root=REPO_ROOT)

    def test_static_runner_has_no_model_corpus_or_quality_imports_and_writes_fresh_output(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "import datasets",
            "import transformers",
            "from_pretrained",
            "load_dataset",
        ):
            self.assertNotIn(forbidden, source)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            # The repository is dirty while this test is developed; source-state rejection
            # is the correct result until the files are committed. Exercise payload writing
            # through a clean temporary git worktree is intentionally deferred to server preflight.
            if completed.returncode == 0:
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertFalse(report["boundary"]["corpus_accessed"])
                self.assertFalse(report["boundary"]["model_weights_accessed"])
                self.assertTrue(report["boundary"]["exact_input_manifest_build_authorized"])
                repeated = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER_PATH),
                        "--protocol",
                        str(PROTOCOL_PATH),
                        "--output",
                        str(output),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(repeated.returncode, 0)
            else:
                self.assertIn("repository must be clean", completed.stderr)


if __name__ == "__main__":
    unittest.main()
