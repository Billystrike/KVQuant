import unittest
from types import SimpleNamespace

from models.cage_config import get_cage_config


class CageConfigTest(unittest.TestCase):
    def test_missing_fields_get_safe_defaults(self):
        config = SimpleNamespace()

        cage_config = get_cage_config(config)

        self.assertFalse(cage_config.cage_enable)
        self.assertEqual(cage_config.cage_mode, "fake")
        self.assertEqual(cage_config.cage_k_group_sizes, [32, 64, 128])
        self.assertEqual(config.cage_v_clip_percentiles, [0.999, 0.995, 0.99])

    def test_enabled_config_accepts_tuple_inputs(self):
        config = SimpleNamespace(
            cage_enable=True,
            cage_k_group_sizes=(16, 32),
            cage_k_clip_percentiles=(0.999, 0.99),
            cage_k_num_buckets=2,
            cage_v_group_sizes=(16,),
            cage_v_clip_percentiles=(0.999,),
            cage_v_num_buckets=1,
        )

        cage_config = get_cage_config(config)

        self.assertEqual(cage_config.cage_k_group_sizes, [16, 32])
        self.assertEqual(config.cage_v_group_sizes, [16])

    def test_unknown_mode_only_errors_when_enabled(self):
        disabled = SimpleNamespace(cage_enable=False, cage_mode="real")

        cage_config = get_cage_config(disabled)

        self.assertEqual(cage_config.cage_mode, "real")

        with self.assertRaisesRegex(ValueError, "Unsupported CAGE mode"):
            get_cage_config(SimpleNamespace(cage_enable=True, cage_mode="real"))

    def test_invalid_bucket_lengths_error_when_enabled(self):
        config = SimpleNamespace(cage_enable=True, cage_k_group_sizes=[32, 64])

        with self.assertRaisesRegex(ValueError, "cage_k_group_sizes must have 3 entries"):
            get_cage_config(config)


if __name__ == "__main__":
    unittest.main()
