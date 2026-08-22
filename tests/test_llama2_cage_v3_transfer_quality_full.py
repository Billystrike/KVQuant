import json
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_quality_acceptance import lf_normalized_file_sha256
from utils.llama2_cage_v3_transfer_quality_full import expand_full_cases


REPO_ROOT = Path(__file__).resolve().parents[1]


class Llama2CageV3TransferQualityFullTest(unittest.TestCase):
    def _design(self):
        return {
            "input_manifest": {"sha256": "i" * 64},
            "case_matrix": {"anchor_indices": list(range(50))},
        }

    def _protocol(self):
        rows = []
        for length in (1024, 2048, 4032):
            rows.append({
                "prompt_length": length,
                "packed_memory": {"candidate_bytes": 1, "cage_v1_bytes": 2, "kivi_bytes": 3},
                "methods": [
                    {"role": "quality_reference", "id": f"fp16-{length}", "method": "fp16"},
                    {"role": "transfer_candidate", "id": f"v3-{length}", "method": "cage_v3"},
                    {"role": "primary_predecessor", "id": f"v1-{length}", "method": "cage_v1"},
                    {"role": "primary_uniform_baseline", "id": f"kivi-{length}", "method": "kivi"},
                ],
            })
        return {"method_length_matrix": rows}

    def _manifest(self):
        inputs = []
        for anchor in range(50):
            for length in (1024, 2048, 4032):
                inputs.append({
                    "input_id": f"{anchor}-{length}",
                    "identity": {"anchor_index": anchor, "prompt_length": length},
                    "prompt_ids": [1] * length,
                    "continuation_ids": [2] * 64,
                })
        return {"inputs": inputs}

    def test_expands_exact_600_cases_in_frozen_point_then_anchor_order(self):
        cases = expand_full_cases(
            design=self._design(),
            protocol=self._protocol(),
            input_manifest=self._manifest(),
            gate_receipt_sha256="g" * 64,
        )
        self.assertEqual(len(cases), 600)
        self.assertEqual(len({case["case_id"] for case in cases}), 600)
        self.assertEqual([case["input"]["identity"]["anchor_index"] for case in cases[:50]], list(range(50)))
        self.assertEqual(cases[0]["method"]["method"], "fp16")
        self.assertEqual(cases[50]["method"]["method"], "cage_v3")
        self.assertEqual(cases[200]["input"]["identity"]["prompt_length"], 2048)

    def test_case_ids_bind_gate_receipt(self):
        first = expand_full_cases(design=self._design(), protocol=self._protocol(), input_manifest=self._manifest(), gate_receipt_sha256="a" * 64)
        second = expand_full_cases(design=self._design(), protocol=self._protocol(), input_manifest=self._manifest(), gate_receipt_sha256="b" * 64)
        self.assertNotEqual(first[0]["case_id"], second[0]["case_id"])

    def test_runner_freezes_determinism_resume_and_no_interpretation(self):
        source = (REPO_ROOT / "scripts" / "llama2_run_cage_v3_transfer_quality_full.py").read_text(encoding="utf-8")
        self.assertLess(source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]'), source.index("import torch"))
        self.assertIn('f"[{index}/600]', source)
        self.assertIn("validate_full_case(existing, case)", source)
        self.assertIn('"quality_interpretation_authorized": False', source)
        self.assertIn('"expected_cases": 600', source)

    def test_gate_hashes_weights_without_loading_them(self):
        source = (REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_quality_full_gate.py").read_text(encoding="utf-8")
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("cross_entropy", source)
        self.assertIn('"no_model_weights_loaded": True', source)
        self.assertIn('"full_600_case_execution_authorized": True', source)

    def test_gate_plan_freezes_every_runtime_source_without_authorizing_execution_early(self):
        path = REPO_ROOT / "configs" / "llama2_7b_cage_v3_transfer_quality_full_gate_plan_v1.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        for spec in plan["frozen_sources"].values():
            self.assertEqual(lf_normalized_file_sha256(REPO_ROOT / spec["path"]), spec["sha256"])
        self.assertTrue(plan["authorization_before_gate"]["full_gate_execution"])
        self.assertFalse(plan["authorization_before_gate"]["full_600_case_execution"])
        self.assertFalse(plan["authorization_before_gate"]["quality_interpretation"])


if __name__ == "__main__":
    unittest.main()
