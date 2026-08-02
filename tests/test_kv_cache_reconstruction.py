import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import torch


def _install_optional_triton_stub():
    if "triton" in sys.modules:
        return
    triton = types.ModuleType("triton")
    triton.__spec__ = ModuleSpec("triton", loader=None)
    triton.jit = lambda fn=None, **_: fn if fn is not None else lambda wrapped: wrapped
    triton_language = types.ModuleType("triton.language")
    triton_language.__spec__ = ModuleSpec("triton.language", loader=None)
    triton_language.constexpr = object()
    triton.language = triton_language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton_language


_install_optional_triton_stub()

from models.cage_cache import CageKeyCache, CageValueCache, pack_cage_past_key_value
from utils.kv_cache_reconstruction import reconstruct_cage_cache, reconstruct_kivi_cache


class KvCacheReconstructionTest(unittest.TestCase):
    def _expected(self):
        expected = torch.zeros(1, 1, 17, 16, dtype=torch.float16)
        expected[:, :, -1, :] = 1.0
        return expected

    def test_reconstructs_actual_kivi_packed_tuple(self):
        zero_codes = torch.zeros(1, 1, 16, 1, dtype=torch.int32)
        unit_scales = torch.ones(1, 1, 16, 1, dtype=torch.float16)
        zero_mins = torch.zeros_like(unit_scales)
        full_suffix = torch.ones(1, 1, 1, 16, dtype=torch.float16)
        past_key_value = (
            zero_codes,
            full_suffix,
            unit_scales,
            zero_mins,
            zero_codes,
            full_suffix,
            unit_scales,
            zero_mins,
            17,
        )

        key, value = reconstruct_kivi_cache(
            past_key_value, group_size=16, k_bits=2, v_bits=2
        )

        self.assertEqual(key.shape, (1, 1, 17, 16))
        self.assertEqual(value.shape, (1, 1, 17, 16))
        torch.testing.assert_close(key, self._expected())
        torch.testing.assert_close(value, self._expected())

    def test_kivi_reconstruction_uses_separate_key_and_value_widths(self):
        code = torch.zeros(1, 1, 16, 1, dtype=torch.int32)
        scale = torch.ones(1, 1, 16, 1, dtype=torch.float16)
        minimum = torch.zeros_like(scale)
        cache = (code, None, scale, minimum, code, None, scale, minimum, 16)

        with patch(
            "quant.new_pack.unpack_and_dequant_vcache",
            return_value=torch.zeros(1, 1, 16, 16, dtype=torch.float16),
        ) as unpack:
            reconstruct_kivi_cache(cache, group_size=16, k_bits=2, v_bits=4)

        self.assertEqual([call.args[-1] for call in unpack.call_args_list], [2, 4])

    def test_kivi_reconstruction_handles_quant_only_and_full_only_histories(self):
        zero_codes = torch.zeros(1, 1, 16, 1, dtype=torch.int32)
        unit_scales = torch.ones(1, 1, 16, 1, dtype=torch.float16)
        zero_mins = torch.zeros_like(unit_scales)
        full = torch.ones(1, 1, 1, 16, dtype=torch.float16)

        quant_key, quant_value = reconstruct_kivi_cache(
            (zero_codes, None, unit_scales, zero_mins, zero_codes, None, unit_scales, zero_mins, 16),
            group_size=16,
            k_bits=2,
            v_bits=2,
        )
        full_key, full_value = reconstruct_kivi_cache(
            (None, full, None, None, None, full, None, None, 1),
            group_size=16,
            k_bits=2,
            v_bits=2,
        )

        torch.testing.assert_close(quant_key, torch.zeros(1, 1, 16, 16, dtype=torch.float16))
        torch.testing.assert_close(quant_value, torch.zeros(1, 1, 16, 16, dtype=torch.float16))
        torch.testing.assert_close(full_key, full)
        torch.testing.assert_close(full_value, full)

    def test_reconstructs_cage_fake_cache(self):
        quant_bucket = torch.zeros(1, 1, 16, 16, dtype=torch.float16)
        full_suffix = torch.ones(1, 1, 1, 16, dtype=torch.float16)
        bucket_indices = (torch.arange(16).view(1, 16),)
        past_key_value = pack_cage_past_key_value(
            key_cache=CageKeyCache(
                key_quant_buckets=(quant_bucket,),
                key_full=full_suffix,
                key_bucket_indices=bucket_indices,
            ),
            value_cache=CageValueCache(
                value_quant_buckets=(quant_bucket,),
                value_full=full_suffix,
                value_bucket_indices=bucket_indices,
            ),
        )

        key, value = reconstruct_cage_cache(past_key_value)

        self.assertEqual(key.shape, (1, 1, 17, 16))
        self.assertEqual(value.shape, (1, 1, 17, 16))
        torch.testing.assert_close(key, self._expected())
        torch.testing.assert_close(value, self._expected())

    def test_cage_scatter_uses_per_head_indices_across_multiple_buckets(self):
        first = torch.ones(1, 2, 2, 8, dtype=torch.float16)
        second = torch.full((1, 2, 2, 8), 2.0, dtype=torch.float16)
        even = torch.arange(0, 16, 2)
        odd = torch.arange(1, 16, 2)
        indices = (torch.stack((even, odd)), torch.stack((odd, even)))
        cache = pack_cage_past_key_value(
            key_cache=CageKeyCache(
                key_quant_buckets=(first, second),
                key_bucket_indices=indices,
            ),
            value_cache=CageValueCache(
                value_quant_buckets=(first, second),
                value_bucket_indices=indices,
            ),
        )
        expected = torch.empty(1, 2, 2, 16, dtype=torch.float16)
        expected[:, 0, :, even] = 1.0
        expected[:, 0, :, odd] = 2.0
        expected[:, 1, :, odd] = 1.0
        expected[:, 1, :, even] = 2.0

        key, value = reconstruct_cage_cache(cache)

        torch.testing.assert_close(key, expected)
        torch.testing.assert_close(value, expected)

    def test_cage_reconstruction_handles_full_only_history(self):
        full = torch.ones(1, 1, 1, 16, dtype=torch.float16)
        cache = pack_cage_past_key_value(
            key_cache=CageKeyCache(key_full=full),
            value_cache=CageValueCache(value_full=full),
        )

        key, value = reconstruct_cage_cache(cache)

        torch.testing.assert_close(key, full)
        torch.testing.assert_close(value, full)


if __name__ == "__main__":
    unittest.main()
