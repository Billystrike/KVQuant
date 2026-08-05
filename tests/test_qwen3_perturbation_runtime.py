import unittest


try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache

    from models.qwen3_cage import install_qwen3_cage_attention
    from models.qwen3_kivi import Qwen3KiviCache, Qwen3KiviCacheConfig
    from utils.qwen3_perturbation_runtime import Qwen3PerturbationRecorder

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


@unittest.skipUnless(QWEN3_AVAILABLE, "requires the frozen Transformers Qwen3 stack")
class Qwen3PerturbationRuntimeTest(unittest.TestCase):
    def setUp(self):
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
        self.model = Qwen3ForCausalLM(config).eval()
        self.assertEqual(install_qwen3_cage_attention(self.model), 2)
        self.recorder = Qwen3PerturbationRecorder(expected_layers=2, top_k=3)
        self.assertEqual(self.recorder.install(self.model), 2)
        self.prompt = torch.tensor([[1, 2, 3, 4, 5]])
        self.decode = torch.tensor([[6]])

    def _run(self, cache):
        self.recorder.begin_reference(prompt_length=5)
        with torch.no_grad():
            self.model(
                input_ids=torch.cat((self.prompt, self.decode), dim=1),
                use_cache=False,
            )
        self.recorder.begin_candidate_prefill()
        with torch.no_grad():
            self.model(input_ids=self.prompt, past_key_values=cache, use_cache=True)
        self.recorder.begin_candidate_decode()
        with torch.no_grad():
            self.model(input_ids=self.decode, past_key_values=cache, use_cache=True)
        return self.recorder.finish()

    def test_fp16_cache_is_an_exact_local_identity(self):
        records = self._run(DynamicCache())
        self.assertEqual([record["layer_idx"] for record in records], [0, 1])
        for record in records:
            for name, value in record["metrics"].items():
                self.assertEqual(value, 1.0 if name == "topk_attention_overlap" else 0.0)

    def test_kivi_cache_produces_finite_nonzero_local_perturbation(self):
        records = self._run(
            Qwen3KiviCache(
                Qwen3KiviCacheConfig(group_size=4, residual_length=4)
            )
        )
        self.assertTrue(
            any(record["metrics"]["relative_k_reconstruction_error"] > 0 for record in records)
        )
        self.assertTrue(
            any(record["metrics"]["relative_v_reconstruction_error"] > 0 for record in records)
        )
        for record in records:
            self.assertTrue(
                all(torch.isfinite(torch.tensor(value)) for value in record["metrics"].values())
            )


if __name__ == "__main__":
    unittest.main()
