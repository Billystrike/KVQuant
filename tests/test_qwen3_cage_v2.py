import unittest
import hashlib
import json
from pathlib import Path

from utils.qwen3_cage_v2 import (
    CageV2LayerOption,
    allocate_layer_options,
    estimate_qwen3_cage_v2_bytes,
)
from utils.qwen3_memory import estimate_qwen3_kivi_bytes, estimate_qwen3_kitty_bytes
from utils.qwen3_cage_v2_protocol import (
    build_cage_v2_dev_manifest,
    validate_cage_v2_dev_manifest,
)


try:
    import torch

    from models.cage_v2_quant import (
        fake_quant_k_sparse_refinement,
        fake_quant_k_uniform,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class CageV2QuantTest(unittest.TestCase):
    def test_sparse_refinement_changes_only_selected_channels_after_int2_base(self):
        states = torch.tensor(
            [[[[0.0, -3.0, 1.0, 4.0], [1.0, -1.0, 2.0, 2.0],
               [4.0, 2.0, -2.0, 0.0], [7.0, 6.0, -1.0, -3.0]]]],
            dtype=torch.float32,
        )
        base = fake_quant_k_uniform(states, group_size=4, bits=2)
        refined = fake_quant_k_sparse_refinement(
            states,
            one_bit_indices=torch.tensor([[1]]),
            two_bit_indices=torch.tensor([[0]]),
            base_group_size=4,
            refinement_group_size=4,
        )

        torch.testing.assert_close(refined[..., 2:], base[..., 2:])
        self.assertLess(
            torch.mean((states[..., :2] - refined[..., :2]).square()).item(),
            torch.mean((states[..., :2] - base[..., :2]).square()).item(),
        )
        self.assertTrue(bool(torch.isfinite(refined).all()))

    def test_sparse_refinement_rejects_overlapping_modes(self):
        states = torch.randn(1, 2, 4, 8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fake_quant_k_sparse_refinement(
                states,
                one_bit_indices=torch.tensor([[1], [1]]),
                two_bit_indices=torch.tensor([[1], [1]]),
                base_group_size=4,
                refinement_group_size=4,
            )


class CageV2MemoryTest(unittest.TestCase):
    def test_zero_refinement_matches_uniform_kivi_layout(self):
        cage = estimate_qwen3_cage_v2_bytes(
            seq_len=1024,
            residual_length=128,
            one_bit_channels=0,
            two_bit_channels=0,
            sink_length=32,
        )
        kivi = estimate_qwen3_kivi_bytes(
            seq_len=1024,
            group_size=128,
            residual_length=128,
            sink_length=32,
        )
        self.assertEqual(cage["model_total_bytes"], kivi["model_total_bytes"])
        self.assertFalse(cage["policy"]["value_adaptive"])

    def test_refinement_bytes_are_monotonic_and_layer_specific(self):
        base = estimate_qwen3_cage_v2_bytes(
            seq_len=2048,
            residual_length=128,
            one_bit_channels=0,
            two_bit_channels=0,
        )
        one_layer = estimate_qwen3_cage_v2_bytes(
            seq_len=2048,
            residual_length=128,
            one_bit_channels=[8] + [0] * 35,
            two_bit_channels=[4] + [0] * 35,
        )
        uniform = estimate_qwen3_cage_v2_bytes(
            seq_len=2048,
            residual_length=128,
            one_bit_channels=8,
            two_bit_channels=4,
        )
        self.assertLess(base["model_total_bytes"], one_layer["model_total_bytes"])
        self.assertLess(one_layer["model_total_bytes"], uniform["model_total_bytes"])
        self.assertGreater(
            one_layer["layer_reports"][0]["components"]["key_refinement_index_bytes"],
            0,
        )
        self.assertEqual(
            one_layer["layer_reports"][1]["components"]["key_refinement_index_bytes"],
            0,
        )
        self.assertEqual(
            one_layer["model_total_bytes"],
            sum(layer["components"]["total_bytes"] for layer in one_layer["layer_reports"]),
        )

    def test_allocator_minimizes_error_under_exact_budget(self):
        options = [
            [
                CageV2LayerOption(0, 0, 100, 10.0, "l0-base"),
                CageV2LayerOption(0, 4, 120, 2.0, "l0-refine"),
            ],
            [
                CageV2LayerOption(0, 0, 100, 10.0, "l1-base"),
                CageV2LayerOption(0, 4, 120, 7.0, "l1-refine"),
            ],
        ]
        selected = allocate_layer_options(options, target_bytes=220)
        self.assertEqual(selected["total_bytes"], 220)
        self.assertEqual(selected["selected_labels"], ["l0-refine", "l1-base"])
        self.assertEqual(selected["two_bit_channels_per_layer"], [4, 0])


class CageV2DevelopmentProtocolTest(unittest.TestCase):
    def _protocol(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "qwen3_8b_cage_v2_feasibility_protocol_draft_v1.json"
        )
        raw = path.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    def test_dev_manifest_uses_separate_claim_ineligible_validation_anchors(self):
        protocol, protocol_sha256 = self._protocol()
        tokens = list(range(100000))
        manifest = build_cage_v2_dev_manifest(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            token_ids=tokens,
            corpus_snapshot_sha256="a" * 64,
            corpus_identity={
                "id": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "split": "validation",
                "revision": "00aa25585682d4957f9e86edc73f59be7419af99",
                "join_separator": "\n\n",
            },
            tokenizer_identity={"class": "test"},
            source_state={"git_commit": "test", "dirty": False},
        )
        self.assertEqual(manifest["status"], "development_only_not_eligible_for_final_claims")
        self.assertEqual(manifest["corpus_identity"]["split"], "validation")
        self.assertEqual(len(manifest["cases"]), 30)
        self.assertGreaterEqual(
            manifest["selection"]["minimum_anchor_gap"],
            manifest["selection"]["maximum_window_tokens"],
        )
        validate_cage_v2_dev_manifest(
            manifest,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )

    def test_protocol_limits_design_rounds_and_excludes_v1_test_grid(self):
        protocol, _ = self._protocol()
        self.assertEqual(protocol["development_limits"]["maximum_design_rounds"], 2)
        self.assertFalse(protocol["development_data"]["eligible_for_final_claims"])
        exclusions = " ".join(protocol["development_data"]["exclusions"])
        self.assertIn("50-anchor", exclusions)
        self.assertFalse(protocol["fixed_quantization_choices"]["value_adaptive"])

    def test_round1_points_are_byte_only_under_budget_and_reproducible(self):
        protocol, _ = self._protocol()
        data = protocol["development_data"]
        self.assertEqual(
            set(data["calibration_anchor_indices"])
            | set(data["round1_screening_anchor_indices"]),
            set(range(10)),
        )
        self.assertFalse(
            set(data["calibration_anchor_indices"])
            & set(data["round1_screening_anchor_indices"])
        )
        points = protocol["round1"]["cage_v2_points"]
        self.assertEqual(len(points), 18)
        self.assertEqual(len({point["method_id"] for point in points}), 18)
        for point in points:
            report = estimate_qwen3_cage_v2_bytes(
                seq_len=point["prompt_length"],
                residual_length=point["residual_length"],
                one_bit_channels=point["one_bit_channels"],
                two_bit_channels=point["two_bit_channels"],
                sink_length=point["sink_length"],
            )
            self.assertEqual(report["model_total_bytes"], point["packed_bytes"])
            self.assertLessEqual(point["packed_bytes"], point["target_bytes"])
            eligible = []
            for residual in range(16, 513, 16):
                candidate = estimate_qwen3_cage_v2_bytes(
                    seq_len=point["prompt_length"],
                    residual_length=residual,
                    one_bit_channels=point["one_bit_channels"],
                    two_bit_channels=point["two_bit_channels"],
                    sink_length=point["sink_length"],
                )["model_total_bytes"]
                if candidate <= point["target_bytes"]:
                    eligible.append(candidate)
            self.assertEqual(point["packed_bytes"], max(eligible))
        for point in protocol["round1"]["kitty_points"]:
            report = estimate_qwen3_kitty_bytes(
                seq_len=point["prompt_length"],
                boosted_channels=point["boosted_channels"],
            )
            self.assertEqual(report["model_total_bytes"], point["packed_bytes"])


try:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache

    from models.qwen3_cage import install_qwen3_cage_attention
    from models.qwen3_cage_v2 import CageV2Config, Qwen3CageV2Cache

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


@unittest.skipUnless(QWEN3_AVAILABLE, "requires the frozen Transformers Qwen3 stack")
class Qwen3CageV2CacheTest(unittest.TestCase):
    def test_prefill_is_exact_and_storage_uses_sparse_key_uniform_value(self):
        torch.manual_seed(11)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            attention_dropout=0.0,
            use_cache=True,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        install_qwen3_cage_attention(model)
        cache = Qwen3CageV2Cache(
            CageV2Config(
                one_bit_channels=1,
                two_bit_channels=2,
                key_base_group_size=4,
                key_refinement_group_size=4,
                value_group_size=4,
                sink_length=1,
            ),
            residual_length=4,
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        with torch.no_grad():
            reference = model(
                input_ids=input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
            candidate = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

        torch.testing.assert_close(candidate.logits, reference.logits)
        self.assertEqual(cache.key_quantized_lengths, [5, 5])
        self.assertEqual(cache.value_quantized_lengths, [3, 3])
        for policy in cache.layer_policies:
            self.assertEqual(tuple(policy.one_bit_indices.shape), (2, 1))
            self.assertEqual(tuple(policy.two_bit_indices.shape), (2, 2))
            for head in range(2):
                self.assertEqual(
                    set(policy.one_bit_indices[head].tolist())
                    & set(policy.two_bit_indices[head].tolist()),
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
