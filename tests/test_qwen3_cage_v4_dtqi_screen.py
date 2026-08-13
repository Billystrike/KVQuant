import copy
import unittest
from pathlib import Path

from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_screen import (
    expand_screen_cases,
    load_screen_execution,
    validate_gpu_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_gpu_acceptance_receipt_v1.json"
EXECUTION = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_execution_v1.json"


def _manifest():
    anchors = []
    cases = []
    token = 0
    for document in range(20):
        for anchor_index in range(2):
            full = list(range(token, token + 4096))
            token += 4096
            compact = []
            for length in (1024, 2048, 4032):
                case_id = f"d{document}-a{anchor_index}-l{length}"
                identity = {
                    "partition": "screen",
                    "document_id": f"document-{document}",
                    "anchor_index": anchor_index,
                    "prompt_length": length,
                }
                cases.append({"case_id": case_id, "identity": identity})
                compact.append({"case_id": case_id, "prompt_length": length})
            anchors.append({
                "partition": "screen",
                "document_id": f"document-{document}",
                "anchor_index": anchor_index,
                "full_window_ids": full,
                "cases": compact,
            })
    return {"anchors": anchors, "cases": cases}


class CageV4DTQIScreenTest(unittest.TestCase):
    def test_gpu_acceptance_receipt_opens_only_full_screen(self):
        receipt = load_json(RECEIPT)
        validate_gpu_receipt(receipt, verify_artifacts=False)
        authorization = receipt["authorization"]
        self.assertTrue(authorization["gpu_full_screen"])
        self.assertFalse(authorization["holdout_method_metrics"])
        self.assertFalse(authorization["pg19_test_access"])
        self.assertFalse(authorization["interpret_before_full_postrun"])

    def test_receipt_rejects_holdout_mutation(self):
        receipt = copy.deepcopy(load_json(RECEIPT))
        receipt["authorization"]["holdout_method_metrics"] = True
        with self.assertRaisesRegex(RuntimeError, "authorization changed"):
            validate_gpu_receipt(receipt, verify_artifacts=False)

    def test_frozen_execution_expands_exact_120_cases(self):
        execution, execution_sha, protocol, _, quota = load_screen_execution(
            EXECUTION, repo_root=REPO_ROOT, verify_artifacts=False
        )
        cases = expand_screen_cases(
            execution=execution,
            execution_sha256=execution_sha,
            protocol=protocol,
            quota_plan=quota,
            input_manifest=_manifest(),
        )
        self.assertEqual(len(cases), 120)
        self.assertEqual(len({case["case_id"] for case in cases}), 120)
        self.assertEqual({case["input"]["prompt_length"] for case in cases}, {1024, 2048, 4032})
        self.assertEqual({case["method"]["candidate_id"] for case in cases}, {"cage-v4-dtqi"})
        self.assertTrue(all(case["method"]["config"]["recent_query_window"] == case["method"]["config"]["residual_length"] for case in cases))
        self.assertTrue(all(sum(case["method"]["config"]["two_bit_channels"]) == 1152 for case in cases))

    def test_screen_runner_exposes_no_partition_or_stage_override(self):
        source = (REPO_ROOT / "scripts" / "qwen3_run_cage_v4_dtqi_screen.py").read_text(encoding="utf-8")
        self.assertNotIn('"--partition"', source)
        self.assertNotIn('"--stage"', source)
        self.assertNotIn("local_perturbation", source)


if __name__ == "__main__":
    unittest.main()
