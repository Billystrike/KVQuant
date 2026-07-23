import unittest

import torch

from models.cage_importance import (
    assign_channel_buckets,
    compute_key_importance,
    compute_value_importance,
)


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

    def test_importance_rejects_non_finite_source_values(self):
        query_states = torch.tensor([[[[float("nan"), 2.0], [3.0, 4.0]]]])
        key_states = torch.tensor([[[[1.0, 5.0], [1.0, 5.0]]]])
        value_states = torch.tensor([[[[2.0, float("nan")], [2.0, 3.0]]]])
        o_proj_weight = torch.ones(1, 2)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_key_importance(query_states, key_states)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_value_importance(
                value_states,
                o_proj_weight,
                num_heads=1,
                num_key_value_heads=1,
                head_dim=2,
            )

    def test_assign_channel_buckets_places_highest_importance_in_first_bucket(self):
        importance = torch.tensor(
            [
                [0.1, 0.9, 0.4, 0.2, 0.8],
                [5.0, 2.0, 3.0, 4.0, 1.0],
            ]
        )

        assignment = assign_channel_buckets(importance, num_buckets=3)

        self.assertEqual(assignment.num_buckets, 3)
        self.assertEqual([tuple(index.shape) for index in assignment.bucket_indices], [(2, 1), (2, 2), (2, 2)])
        self.assertTrue(torch.equal(assignment.bucket_indices[0], torch.tensor([[1], [0]])))
        self.assertTrue(torch.equal(assignment.bucket_rank_map, torch.tensor([[2, 0, 1, 2, 1], [0, 2, 1, 1, 2]])))

    def test_assign_channel_buckets_reduces_batch_importance_before_assignment(self):
        importance = torch.tensor(
            [
                [[1.0, 4.0, 2.0]],
                [[3.0, 0.0, 6.0]],
            ]
        )

        assignment = assign_channel_buckets(importance, num_buckets=2)

        self.assertTrue(torch.equal(assignment.bucket_indices[0], torch.tensor([[2]])))
        self.assertTrue(torch.equal(assignment.bucket_indices[1], torch.tensor([[0, 1]])))
        self.assertTrue(torch.equal(assignment.bucket_rank_map, torch.tensor([[1, 1, 0]])))

    def test_assign_channel_buckets_breaks_ties_by_channel_index(self):
        importance = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        first = assign_channel_buckets(importance, num_buckets=2)
        second = assign_channel_buckets(importance, num_buckets=2)

        self.assertTrue(torch.equal(first.bucket_indices[0], torch.tensor([[0, 1]])))
        self.assertTrue(torch.equal(first.bucket_indices[1], torch.tensor([[2, 3]])))
        self.assertTrue(torch.equal(first.bucket_rank_map, second.bucket_rank_map))

    def test_assign_channel_buckets_reduces_effective_bucket_count_when_head_dim_is_smaller(self):
        importance = torch.tensor([[0.2, 0.8]])

        assignment = assign_channel_buckets(importance, num_buckets=3)

        self.assertEqual(assignment.num_buckets, 2)
        self.assertEqual(len(assignment.bucket_indices), 2)
        self.assertTrue(torch.equal(assignment.bucket_indices[0], torch.tensor([[1]])))
        self.assertTrue(torch.equal(assignment.bucket_indices[1], torch.tensor([[0]])))

    def test_assign_channel_buckets_can_pad_bucket_indices_and_report_valid_counts(self):
        importance = torch.tensor([[0.1, 0.9, 0.4, 0.2, 0.8]])

        assignment = assign_channel_buckets(importance, num_buckets=3, pad_to_multiple=4)

        self.assertEqual([tuple(index.shape) for index in assignment.bucket_indices], [(1, 4), (1, 4), (1, 4)])
        self.assertEqual(assignment.valid_channel_counts, (1, 2, 2))
        self.assertTrue(torch.equal(assignment.bucket_indices[0][:, :1], torch.tensor([[1]])))
        self.assertTrue(torch.equal(assignment.bucket_indices[1][:, :2], torch.tensor([[4, 2]])))
        self.assertTrue(torch.equal(assignment.bucket_indices[2][:, :2], torch.tensor([[3, 0]])))


if __name__ == "__main__":
    unittest.main()
