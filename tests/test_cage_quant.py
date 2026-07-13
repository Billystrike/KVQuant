import unittest

import torch

from models.cage_quant import (
    fake_quant_k_by_channel_buckets,
    fake_quant_v_by_channel_buckets,
)


class CageQuantTest(unittest.TestCase):
    def test_key_fake_quant_preserves_shape_and_scatters_selected_channels(self):
        key_states = torch.tensor(
            [[[[10.0, 0.0, 20.0], [11.0, 1.4, 21.0], [12.0, 2.7, 22.0], [13.0, 4.0, 23.0]]]]
        )
        bucket_indices = (torch.tensor([[1]]),)

        quantized = fake_quant_k_by_channel_buckets(
            key_states,
            bucket_indices=bucket_indices,
            group_sizes=(4,),
            clip_percentiles=(1.0,),
            bits=2,
        )

        expected_selected = torch.tensor([0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0])
        self.assertEqual(quantized.shape, key_states.shape)
        self.assertTrue(torch.equal(quantized[..., 0], key_states[..., 0]))
        self.assertTrue(torch.equal(quantized[..., 2], key_states[..., 2]))
        self.assertTrue(torch.allclose(quantized[0, 0, :, 1], expected_selected))

    def test_value_fake_quant_preserves_shape_and_quantizes_selected_channels_per_token(self):
        value_states = torch.tensor([[[[0.0, 1.4, 2.7, 4.0, 99.0]]]])
        bucket_indices = (torch.tensor([[0, 1, 2, 3]]),)

        quantized = fake_quant_v_by_channel_buckets(
            value_states,
            bucket_indices=bucket_indices,
            group_sizes=(4,),
            clip_percentiles=(1.0,),
            bits=2,
        )

        expected_selected = torch.tensor([0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0])
        self.assertEqual(quantized.shape, value_states.shape)
        self.assertTrue(torch.allclose(quantized[0, 0, 0, :4], expected_selected))
        self.assertTrue(torch.equal(quantized[..., 4], value_states[..., 4]))

    def test_fake_quant_accepts_one_dimensional_bucket_indices_for_all_heads(self):
        key_states = torch.tensor(
            [
                [
                    [[0.0, 10.0], [1.4, 11.0], [2.7, 12.0], [4.0, 13.0]],
                    [[4.0, 20.0], [2.7, 21.0], [1.4, 22.0], [0.0, 23.0]],
                ]
            ]
        )

        quantized = fake_quant_k_by_channel_buckets(
            key_states,
            bucket_indices=(torch.tensor([0]),),
            group_sizes=(4,),
            clip_percentiles=(1.0,),
        )

        self.assertTrue(torch.equal(quantized[..., 1], key_states[..., 1]))
        self.assertTrue(torch.allclose(quantized[0, 0, :, 0], torch.tensor([0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0])))
        self.assertTrue(torch.allclose(quantized[0, 1, :, 0], torch.tensor([4.0, 8.0 / 3.0, 4.0 / 3.0, 0.0])))

    def test_fake_quant_rejects_non_finite_source_values(self):
        key_states = torch.tensor([[[[float("nan"), float("inf")], [float("-inf"), 1.0]]]])

        with self.assertRaisesRegex(ValueError, "non-finite"):
            fake_quant_k_by_channel_buckets(
                key_states,
                bucket_indices=(torch.tensor([[0, 1]]),),
                group_sizes=(2,),
                clip_percentiles=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()
