from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.llama2_cage_v3_gpu_acceptance_protocol import (
    Llama2CageV3GPUProtocolError,
    compact_ids_sha256,
    load_protocol,
    query_gpu_inventory,
    synthetic_token_ids,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_gpu_acceptance_protocol_v1.json"


class Llama2CageV3GPUAcceptanceProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_checked_in_protocol_freezes_cpu_pass_before_gpu_execution(self):
        protocol, digest = load_protocol(PROTOCOL_PATH, repo_root=REPO_ROOT)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(digest, "d4e3e0468f0719f00632ddc6e0b40e04cc97c459fa2352cd9fe27cb8585273ba")
        self.assertTrue(protocol["cpu_acceptance_receipt"]["cpu_implementation_acceptance_pass"])
        self.assertFalse(protocol["execution_boundary"]["gpu_acceptance_execution_authorized_by_this_protocol_alone"])

    def test_synthetic_input_is_corpus_free_and_hash_exact(self):
        ids = synthetic_token_ids()
        self.assertEqual(len(ids), 1026)
        self.assertEqual(compact_ids_sha256(ids[:1024]), "153a3b94c632846e3414bca42a8e55079b101acf10a61a2489b49a434871482d")
        self.assertEqual(compact_ids_sha256(ids[1024:]), "70575b1540d938978c3f7ddf36c2bb6263e126ede049adf20bd8d1c21119729d")

    def test_protocol_rejects_candidate_metric_or_execution_mutation(self):
        mutated = copy.deepcopy(self.protocol)
        mutated["candidate"]["residual_length"] = 160
        with self.assertRaisesRegex(Llama2CageV3GPUProtocolError, "point"):
            validate_protocol(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.protocol)
        mutated["acceptance_design"]["quality_metric_computed"] = True
        with self.assertRaisesRegex(Llama2CageV3GPUProtocolError, "quality metric"):
            validate_protocol(mutated, repo_root=REPO_ROOT)
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_boundary"]["gpu_acceptance_execution_authorized_by_this_protocol_alone"] = True
        with self.assertRaisesRegex(Llama2CageV3GPUProtocolError, "gpu_acceptance_execution"):
            validate_protocol(mutated, repo_root=REPO_ROOT)

    def test_gpu_inventory_parser_is_read_only_and_exact(self):
        completed = type("Completed", (), {"stdout": "NVIDIA GeForce RTX 4090 D, 24564, 570.00\n"})()
        with patch("subprocess.run", return_value=completed) as mocked:
            inventory = query_gpu_inventory()
        self.assertEqual(inventory["name"], "NVIDIA GeForce RTX 4090 D")
        self.assertEqual(inventory["memory_total_mib"], 24564)
        command = mocked.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")

    def test_static_preflight_does_not_import_torch_or_load_weights(self):
        source = (REPO_ROOT / "scripts" / "llama2_validate_cage_v3_gpu_preflight.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn('.to("cuda")', source)
        self.assertIn("nvidia-smi", (REPO_ROOT / "utils" / "llama2_cage_v3_gpu_acceptance_protocol.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
