import unittest

import torch

from models.cage_cache import (
    CAGE_PAST_KEY_VALUE_TAG,
    CageKeyCache,
    CageValueCache,
    is_cage_past_key_value,
    move_bucket_indices_like,
    pack_cage_past_key_value,
    unpack_cage_past_key_value,
)


class CageCacheTest(unittest.TestCase):
    def _key_cache(self):
        return CageKeyCache(
            key_quant_buckets=(torch.zeros(1, 2, 3, 2), torch.ones(1, 2, 3, 2)),
            key_full=torch.full((1, 2, 1, 4), 2.0),
            key_bucket_indices=(torch.tensor([0, 2]), torch.tensor([1, 3])),
            key_group_sizes=(32, 64),
            key_clip_percentiles=(0.999, 0.99),
            key_scales=(torch.ones(1, 2, 3, 2), None),
            key_mins=(torch.zeros(1, 2, 3, 2), None),
        )

    def _value_cache(self):
        return CageValueCache(
            value_quant_buckets=(torch.zeros(1, 2, 3, 2), torch.ones(1, 2, 3, 2)),
            value_full=torch.full((1, 2, 1, 4), 3.0),
            value_bucket_indices=(torch.tensor([0, 2]), torch.tensor([1, 3])),
            value_group_sizes=(32, 64),
            value_clip_percentiles=(0.999, 0.99),
            value_scales=(torch.ones(1, 2, 3, 2), None),
            value_mins=(torch.zeros(1, 2, 3, 2), None),
        )

    def test_cage_cache_round_trips_through_packed_tuple(self):
        key_cache = self._key_cache()
        value_cache = self._value_cache()

        packed = pack_cage_past_key_value(
            key_cache=key_cache,
            value_cache=value_cache,
        )
        unpacked = unpack_cage_past_key_value(packed)

        self.assertTrue(is_cage_past_key_value(packed))
        self.assertEqual(packed[0], CAGE_PAST_KEY_VALUE_TAG)
        self.assertEqual(packed[-1], 4)
        self.assertIs(unpacked.key_cache, key_cache)
        self.assertIs(unpacked.value_cache, value_cache)
        self.assertEqual(unpacked.kv_seq_len, 4)

    def test_original_kivi_tuple_is_not_cage_cache(self):
        original_kivi_cache = (None, None, None, None, None, None, None, None, 8)

        self.assertFalse(is_cage_past_key_value(original_kivi_cache))
        with self.assertRaisesRegex(ValueError, "not a CAGE past_key_value"):
            unpack_cage_past_key_value(original_kivi_cache)

    def test_explicit_kv_seq_len_is_preserved_when_shapes_are_unavailable(self):
        key_cache = CageKeyCache()
        value_cache = CageValueCache()

        packed = pack_cage_past_key_value(
            key_cache=key_cache,
            value_cache=value_cache,
            kv_seq_len=7,
        )
        unpacked = unpack_cage_past_key_value(packed)

        self.assertEqual(packed[-1], 7)
        self.assertEqual(unpacked.kv_seq_len, 7)

    def test_pack_rejects_mismatched_explicit_kv_seq_len(self):
        with self.assertRaisesRegex(ValueError, "kv_seq_len"):
            pack_cage_past_key_value(
                key_cache=self._key_cache(),
                value_cache=self._value_cache(),
                kv_seq_len=5,
            )

    def test_bucket_indices_can_be_moved_to_payload_device(self):
        target = torch.empty(1, device="cpu")
        indices = (torch.tensor([0, 2]), torch.tensor([1, 3]))

        moved = move_bucket_indices_like(indices, target)

        self.assertEqual(len(moved), 2)
        self.assertTrue(all(index.device == target.device for index in moved))


if __name__ == "__main__":
    unittest.main()
