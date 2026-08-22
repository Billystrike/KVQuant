import copy
import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_acceptance import lf_normalized_file_sha256
from utils.llama2_cage_v3_transfer_quality_full_design import (
    DESIGN_SHA256,
    validate_design,
    validate_postrun_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_full_design_v1.json"
RECEIPT_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_postrun_receipt_v1.json"


class Llama2CageV3TransferQualityFullDesignTest(unittest.TestCase):
    def test_checked_in_design_and_postrun_receipt_are_frozen(self):
        design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(lf_normalized_file_sha256(DESIGN_PATH), DESIGN_SHA256)
        validate_postrun_receipt(receipt, repo_root=REPO_ROOT)
        validate_design(design, repo_root=REPO_ROOT)
        self.assertEqual(design["case_matrix"]["full_case_count"], 600)
        self.assertEqual(design["case_matrix"]["total_target_count"], 38400)

    def test_design_rejects_case_scoring_or_authorization_mutation(self):
        design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(design)
        changed["case_matrix"]["full_case_count"] = 599
        mutations.append(changed)
        changed = copy.deepcopy(design)
        changed["scoring"]["target_count_per_case"] = 63
        mutations.append(changed)
        changed = copy.deepcopy(design)
        changed["authorization"]["full_600_case_execution"] = True
        mutations.append(changed)
        changed = copy.deepcopy(design)
        changed["analysis_boundary"]["read_or_interpret_partial_results"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                validate_design(mutation, repo_root=REPO_ROOT)

    def test_postrun_receipt_does_not_prematurely_authorize_runner_or_full_execution(self):
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(receipt["decision"]["full_execution_design_preflight_authorized"])
        self.assertFalse(receipt["decision"]["full_runner_implementation_authorized"])
        self.assertFalse(receipt["decision"]["full_execution_gate_authorized"])
        self.assertFalse(receipt["decision"]["full_600_case_execution_authorized"])

    def test_static_preflight_never_loads_model_or_imports_torch(self):
        source = (REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_quality_full_design.py").read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("cross_entropy", source)
        self.assertIn('"model_weights_accessed": False', source)
        self.assertIn('"full_600_case_execution_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
