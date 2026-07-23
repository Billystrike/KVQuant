import json
import tempfile
import unittest
from pathlib import Path

from utils.cage_experiment_config import expand_jobs, load_and_resolve_manifest


class CageExperimentConfigTests(unittest.TestCase):
    def _manifest(self):
        return {
            "model": {
                "reference": "/models/llama",
                "dtype": "float16",
                "device": "cuda",
                "max_position_embeddings": 4096,
            },
            "prompts_file": "/data/prompts.jsonl",
            "sample_ids": ["doc-001", "doc-002"],
            "prompt_lengths": [512, 4095],
            "methods": [
                {"id": "fp16", "method": "fp16"},
                {
                    "id": "kivi-g32-r32", "method": "kivi", "k_bits": 2,
                    "v_bits": 2, "group_size": 32, "residual_length": 32,
                },
                {"id": "cage-r32", "method": "cage", "residual_length": 32},
            ],
            "measurement": {"decode_tokens": 1, "seed": 0},
            "output_dir": "/output",
        }

    def _load(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return load_and_resolve_manifest(path)

    def test_resolves_defaults_and_expands_three_deterministic_jobs(self):
        resolved = self._load(self._manifest())
        jobs = expand_jobs(resolved)
        self.assertEqual([job["job_id"] for job in jobs], ["fp16", "kivi-g32-r32", "cage-r32"])
        self.assertEqual(jobs[1]["method_config"], {
            "k_bits": 2, "v_bits": 2, "group_size": 32, "residual_length": 32,
        })
        self.assertEqual(jobs[2]["method_config"], {
            "k_bits": 2, "v_bits": 2, "residual_length": 32,
            "cage_mode": "fake", "cage_k_enable": True, "cage_v_enable": True,
            "cage_k_importance": "q2_var", "cage_k_group_sizes": [32, 64, 128],
            "cage_k_clip_percentiles": [0.999, 0.995, 0.99], "cage_k_num_buckets": 3,
            "cage_v_importance": "wo_var", "cage_v_group_sizes": [32, 64, 128],
            "cage_v_clip_percentiles": [0.999, 0.995, 0.99], "cage_v_num_buckets": 3,
        })

    def test_rejects_unknown_method(self):
        manifest = self._manifest()
        manifest["methods"] = [{"id": "bad", "method": "magic"}]
        with self.assertRaisesRegex(ValueError, "unsupported method"):
            self._load(manifest)

    def test_accepts_prompt_plus_query_at_context_limit(self):
        manifest = self._manifest()
        manifest["prompt_lengths"] = [4095]
        self.assertEqual(self._load(manifest)["prompt_lengths"], [4095])

    def test_rejects_prompt_plus_query_above_context_limit(self):
        manifest = self._manifest()
        manifest["prompt_lengths"] = [4096]
        with self.assertRaisesRegex(ValueError, "context limit.*max_position_embeddings"):
            self._load(manifest)

    def test_rejects_duplicate_prompt_lengths(self):
        manifest = self._manifest()
        manifest["prompt_lengths"] = [512, 512]
        with self.assertRaisesRegex(ValueError, "prompt_lengths.*unique"):
            self._load(manifest)

    def test_rejects_duplicate_resolved_scientific_method_configurations(self):
        manifest = self._manifest()
        duplicate = json.loads(json.dumps(manifest["methods"][2]))
        duplicate["id"] = "same-cage-different-id"
        manifest["methods"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "scientifically duplicate"):
            self._load(manifest)

    def test_checked_manifests_have_frozen_job_point_and_overlap_counts(self):
        root = Path(__file__).parents[1]
        full = load_and_resolve_manifest(root / "configs" / "cage_pilot_llama2_7b.json")
        acceptance = load_and_resolve_manifest(
            root / "configs" / "cage_pilot_llama2_7b_acceptance.json"
        )

        def points(manifest):
            return {
                (
                    method["method"],
                    json.dumps(method["method_config"], sort_keys=True, separators=(",", ":")),
                    sample_id,
                    prompt_length,
                )
                for method in manifest["methods"]
                for sample_id in manifest["sample_ids"]
                for prompt_length in manifest["prompt_lengths"]
            }

        full_points = points(full)
        acceptance_points = points(acceptance)
        self.assertEqual(len(expand_jobs(full)), 10)
        self.assertEqual(len(full_points), 120)
        self.assertEqual(len(expand_jobs(acceptance)), 3)
        self.assertEqual(len(acceptance_points), 6)
        self.assertEqual(len(full_points & acceptance_points), 6)
        self.assertEqual(len(full_points | acceptance_points), 120)
        self.assertEqual(full["prompt_lengths"], [512, 1024, 2048, 4095])

    def test_rejects_invalid_kivi_pair(self):
        manifest = self._manifest()
        manifest["methods"] = [{
            "id": "bad", "method": "kivi", "k_bits": 2, "v_bits": 2,
            "group_size": 64, "residual_length": 32,
        }]
        with self.assertRaisesRegex(ValueError, "residual_length.*group_size"):
            self._load(manifest)

    def test_strictly_rejects_unknown_fields_and_invalid_common_values(self):
        cases = []
        top = self._manifest(); top["surprise"] = True; cases.append(top)
        method = self._manifest(); method["methods"][0]["bits"] = 2; cases.append(method)
        duplicate = self._manifest(); duplicate["methods"][1]["id"] = "fp16"; cases.append(duplicate)
        sample_duplicate = self._manifest(); sample_duplicate["sample_ids"] = ["x", "x"]; cases.append(sample_duplicate)
        bad_length = self._manifest(); bad_length["prompt_lengths"] = [0]; cases.append(bad_length)
        bad_decode = self._manifest(); bad_decode["measurement"]["decode_tokens"] = 2; cases.append(bad_decode)
        bad_dtype = self._manifest(); bad_dtype["model"]["dtype"] = "float32"; cases.append(bad_dtype)
        bad_device = self._manifest(); bad_device["model"]["device"] = "tpu"; cases.append(bad_device)
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self._load(manifest)

    def test_unhashable_scalar_values_raise_value_error(self):
        mutations = [
            ("dtype-list", lambda manifest: manifest["model"].update(dtype=[])),
            ("device-dict", lambda manifest: manifest["model"].update(device={})),
            ("method-list", lambda manifest: manifest["methods"][0].update(method=[])),
            ("k-bits-list", lambda manifest: manifest["methods"][1].update(k_bits=[])),
            ("v-bits-dict", lambda manifest: manifest["methods"][1].update(v_bits={})),
            ("cage-k-bits-dict", lambda manifest: manifest["methods"][2].update(k_bits={})),
        ]
        for name, mutate in mutations:
            manifest = self._manifest()
            mutate(manifest)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._load(manifest)

    def test_rejects_nested_unknown_and_missing_fields(self):
        cases = []
        unknown_model = self._manifest(); unknown_model["model"]["revision"] = "main"; cases.append(unknown_model)
        missing_model = self._manifest(); del missing_model["model"]["reference"]; cases.append(missing_model)
        unknown_measurement = self._manifest(); unknown_measurement["measurement"]["warmup"] = 1; cases.append(unknown_measurement)
        missing_measurement = self._manifest(); del missing_measurement["measurement"]["seed"]; cases.append(missing_measurement)
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self._load(manifest)

    def test_rejects_invalid_cage_bucket_and_clip_settings(self):
        mutations = [
            lambda cage: cage.update(cage_k_group_sizes=[32, 0, 128]),
            lambda cage: cage.update(cage_v_clip_percentiles=[0.999, 0, 0.99]),
            lambda cage: cage.update(cage_k_num_buckets=2),
            lambda cage: cage.update(cage_v_clip_percentiles=[0.999, 1.01, 0.99]),
        ]
        for mutate in mutations:
            manifest = self._manifest()
            mutate(manifest["methods"][2])
            with self.subTest(method=manifest["methods"][2]), self.assertRaises(ValueError):
                self._load(manifest)

    def test_rejects_cage_values_outside_scoped_core_pilot(self):
        mutations = [
            lambda cage: cage.update(k_bits=4),
            lambda cage: cage.update(v_bits=4),
            lambda cage: cage.update(cage_k_enable=False),
            lambda cage: cage.update(cage_v_enable=False),
            lambda cage: cage.update(cage_k_importance="variance"),
            lambda cage: cage.update(cage_v_importance="variance"),
        ]
        for mutate in mutations:
            manifest = self._manifest()
            mutate(manifest["methods"][2])
            with self.subTest(method=manifest["methods"][2]), self.assertRaises(ValueError):
                self._load(manifest)


if __name__ == "__main__":
    unittest.main()
