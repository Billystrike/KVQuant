import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.llama2_cage_v3_transfer_protocol import (
    Llama2CageV3TransferProtocolError,
    audit_llama2_model_metadata,
    derive_llama2_quota_plan,
    derive_packed_memory_preflight,
    load_transfer_protocol,
    validate_transfer_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "llama2_7b_cage_v3_cross_arch_transfer_protocol_v1.json"
QWEN_PLAN_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_calibrated_quota_plan_v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "llama2_validate_cage_v3_transfer_preflight.py"


def _fake_model(root: Path) -> Path:
    model = root / "Llama-2-7b-hf"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "torch_dtype": "float16",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text('{"tokenizer_class":"LlamaTokenizer"}', encoding="utf-8")
    (model / "tokenizer.model").write_bytes(b"fake-tokenizer")
    (model / "pytorch_model-00001-of-00002.bin").write_bytes(b"first-shard")
    (model / "pytorch_model-00002-of-00002.bin").write_bytes(b"second-shard")
    (model / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "pytorch_model-00001-of-00002.bin",
                    "model.layers.31.weight": "pytorch_model-00002-of-00002.bin",
                }
            }
        ),
        encoding="utf-8",
    )
    return model


class Llama2CageV3TransferProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_sha256 = load_transfer_protocol(
            PROTOCOL_PATH, repo_root=REPO_ROOT
        )
        self.qwen_plan = json.loads(QWEN_PLAN_PATH.read_text(encoding="utf-8"))

    def test_preserves_qwen_failure_and_discloses_post_result_followup(self):
        outcome = self.protocol["original_promotion_outcome"]
        disclosure = self.protocol["protocol_deviation_disclosure"]
        self.assertFalse(outcome["promotion_pass"])
        self.assertFalse(outcome["kitty_pro_gate_pass"])
        self.assertEqual(outcome["kitty_pro_4032_relative_ppl_percent"], 1.1904321873860653)
        self.assertTrue(disclosure["original_stop_rule_would_not_authorize_llama2_v3"])
        self.assertTrue(disclosure["followup_was_defined_after_observing_qwen3_promotion_results"])
        self.assertTrue(disclosure["followup_is_frozen_before_any_llama2_v3_metric"])

    def test_depth_mapping_is_deterministic_and_uses_no_llama_metric(self):
        first = derive_llama2_quota_plan(protocol=self.protocol, qwen_plan=self.qwen_plan)
        second = derive_llama2_quota_plan(protocol=self.protocol, qwen_plan=self.qwen_plan)
        self.assertEqual(first, second)
        self.assertFalse(first["llama2_metrics_consumed"])
        self.assertEqual([row["prompt_length"] for row in first["plans"]], [1024, 2048, 4032])
        expected_source_indices = [min(35, int(36 * ((layer + 0.5) / 32))) for layer in range(32)]
        for row in first["plans"]:
            self.assertEqual(row["source_layer_indices"], expected_source_indices)
            self.assertEqual(row["quota_counts"], {"16": 11, "32": 10, "48": 11})
            self.assertEqual(row["quota_total"], 1024)

    def test_byte_only_grid_freezes_exact_predecessor_and_conservative_kivi_points(self):
        quota = derive_llama2_quota_plan(protocol=self.protocol, qwen_plan=self.qwen_plan)
        memory = derive_packed_memory_preflight(protocol=self.protocol, quota_plan=quota)
        expected = {
            1024: (163020800, "cage-r237", 163106816, "kivi-g64-r320", 168820736),
            2048: (249069568, "cage-r253", 248725504, "kivi-g32-r224", 255852544),
            4032: (391348224, "cage-r96", 391254016, "kivi-g128-r256", 398196736),
        }
        self.assertFalse(memory["quality_metrics_consumed"])
        for row in memory["points"]:
            candidate, cage_id, cage_bytes, kivi_id, kivi_bytes = expected[row["prompt_length"]]
            self.assertEqual(row["candidate"]["packed_bytes"], candidate)
            self.assertEqual((row["cage_v1_closest"]["point_id"], row["cage_v1_closest"]["packed_bytes"]), (cage_id, cage_bytes))
            self.assertEqual((row["kivi_closest_feasible"]["point_id"], row["kivi_closest_feasible"]["packed_bytes"]), (kivi_id, kivi_bytes))
            self.assertLessEqual(abs(row["cage_v1_closest"]["relative_byte_delta_vs_candidate"]), 0.0025)
            self.assertGreaterEqual(row["kivi_closest_feasible"]["relative_byte_delta_vs_candidate"], 0.0)
            self.assertFalse(row["kivi_closest_feasible"]["equal_memory_claim_allowed"])

    def test_rejects_result_candidate_grid_or_authorization_mutation(self):
        mutations = []
        changed = copy.deepcopy(self.protocol)
        changed["original_promotion_outcome"]["promotion_pass"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["protocol_deviation_disclosure"]["original_stop_rule_would_not_authorize_llama2_v3"] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["architecture_normalization"]["target_quota_rule"]["top_ranked_11_layers"] = 64
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["frozen_packed_memory_preflight"]["cage_v1_grid"]["residual_length_max"] = 1024
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["kitty_secondary_track"]["included_in_primary_transfer_decision"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.protocol)
        changed["staged_execution"]["stage_3_full_transfer"]["current_authorized"] = True
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(Llama2CageV3TransferProtocolError):
                    validate_transfer_protocol(changed, repo_root=REPO_ROOT)

    def test_model_audit_hashes_tokenizer_metadata_and_every_weight_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = _fake_model(Path(temporary))
            receipt = audit_llama2_model_metadata(model)
            self.assertTrue(all(receipt["checks"].values()))
            self.assertEqual(receipt["weight_file_count"], 2)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in receipt["weight_files"]))
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            config["num_hidden_layers"] = 31
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(Llama2CageV3TransferProtocolError, "architecture"):
                audit_llama2_model_metadata(model)

    def test_static_runner_writes_fresh_artifacts_without_gpu_or_quality_imports(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("mean_nll", source)
        self.assertNotIn("case_files", source)
        self.assertIn('"gpu_used": False', source)
        self.assertIn('"full_transfer_authorized": False', source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _fake_model(root)
            quota = root / "quota.json"
            output = root / "preflight.json"
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--model",
                    str(model),
                    "--quota-output",
                    str(quota),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["boundary"]["gpu_used"])
            self.assertFalse(report["boundary"]["full_transfer_authorized"])
            self.assertEqual(json.loads(quota.read_text(encoding="utf-8"))["plan_id"], "llama2-7b-cage-v3-depth-normalized-quota-plan-v1")
            repeated = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--model",
                    str(model),
                    "--quota-output",
                    str(quota),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("refusing to overwrite", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
