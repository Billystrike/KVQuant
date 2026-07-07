import unittest

import torch

from models.cage_importance import compute_key_importance, compute_value_importance


class CageImportanceTest(unittest.TestCase):
    def test_key_importance_uses_query_energy_and_key_variance_for_mha(self):
        query_states = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 1.0], [2.0, 3.0]],
                ]
            ]
        )
        key_states = torch.tensor(
            [
                [
                    [[1.0, 5.0], [3.0, 5.0]],
                    [[4.0, 1.0], [4.0, 5.0]],
                ]
            ]
        )

        importance = compute_key_importance(query_states, key_states)

        expected = torch.tensor([[5.0, 0.0], [0.0, 20.0]])
        self.assertTrue(torch.allclose(importance, expected))

    def test_key_importance_groups_query_heads_for_gqa(self):
        query_states = torch.tensor(
            [
                [
                    [[1.0], [3.0]],
                    [[2.0], [4.0]],
                    [[1.0], [1.0]],
                    [[3.0], [5.0]],
                ]
            ]
        )
        key_states = torch.tensor([[[[1.0], [3.0]], [[2.0], [6.0]]]])

        importance = compute_key_importance(
            query_states,
            key_states,
            num_key_value_groups=2,
        )

        expected = torch.tensor([[15.0], [72.0]])
        self.assertTrue(torch.allclose(importance, expected))

    def test_reduce_batch_false_keeps_batch_dimension(self):
        query_states = torch.ones(2, 1, 2, 1)
        key_states = torch.tensor([[[[1.0], [3.0]]], [[[2.0], [6.0]]]])

        per_batch = compute_key_importance(query_states, key_states, reduce_batch=False)
        reduced = compute_key_importance(query_states, key_states, reduce_batch=True)

        self.assertEqual(per_batch.shape, (2, 1, 1))
        self.assertTrue(torch.allclose(per_batch, torch.tensor([[[1.0]], [[4.0]]])))
        self.assertTrue(torch.allclose(reduced, per_batch.mean(dim=0)))

    def test_value_importance_uses_output_projection_norm_and_value_variance(self):
        value_states = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 2.0]],
                    [[0.0, 5.0], [4.0, 7.0]],
                ]
            ]
        )
        o_proj_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

        importance = compute_value_importance(
            value_states,
            o_proj_weight,
            num_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        )

        expected = torch.tensor([[10.0, 0.0], [296.0, 100.0]])
        self.assertTrue(torch.allclose(importance, expected))

    def test_value_reduce_batch_false_keeps_batch_dimension(self):
        value_states = torch.tensor([[[[1.0], [3.0]]], [[[2.0], [6.0]]]])
        o_proj_weight = torch.tensor([[2.0]])

        per_batch = compute_value_importance(
            value_states,
            o_proj_weight,
            num_heads=1,
            num_key_value_heads=1,
            head_dim=1,
            reduce_batch=False,
        )
        reduced = compute_value_importance(
            value_states,
            o_proj_weight,
            num_heads=1,
            num_key_value_heads=1,
            head_dim=1,
            reduce_batch=True,
        )

        self.assertEqual(per_batch.shape, (2, 1, 1))
        self.assertTrue(torch.allclose(per_batch, torch.tensor([[[4.0]], [[16.0]]])))
        self.assertTrue(torch.allclose(reduced, per_batch.mean(dim=0)))

    def test_importance_outputs_are_finite_non_negative_and_zero_for_zero_variance(self):
        query_states = torch.tensor([[[[float("nan"), 2.0], [3.0, 4.0]]]])
        key_states = torch.tensor([[[[1.0, 5.0], [1.0, 5.0]]]])
        value_states = torch.tensor([[[[2.0, float("nan")], [2.0, 3.0]]]])
        o_proj_weight = torch.ones(1, 2)

        key_importance = compute_key_importance(query_states, key_states)
        value_importance = compute_value_importance(
            value_states,
            o_proj_weight,
            num_heads=1,
            num_key_value_heads=1,
            head_dim=2,
        )

        self.assertTrue(torch.isfinite(key_importance).all())
        self.assertTrue(torch.isfinite(value_importance).all())
        self.assertTrue((key_importance >= 0).all())
        self.assertTrue((value_importance >= 0).all())
        self.assertTrue(torch.equal(key_importance, torch.zeros_like(key_importance)))


if __name__ == "__main__":
    unittest.main()
