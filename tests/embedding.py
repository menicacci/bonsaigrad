import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Embedding, Module


class TestEmbedding(unittest.TestCase):

    def test_initializes_trainable_lookup_table(self):
        default = Embedding(5, 3)
        first = Embedding(5, 3, rng=np.random.default_rng(7))
        second = Embedding(5, 3, rng=np.random.default_rng(7))

        self.assertEqual(default.weight.data.shape, (5, 3))
        self.assertEqual(first.weight.data.shape, (5, 3))
        self.assertIsInstance(first, Module)
        self.assertEqual(first.parameters(), (first.weight,))
        np.testing.assert_array_equal(first.weight.data, second.weight.data)

    def test_looks_up_vectors_for_every_index_shape(self):
        embedding = Embedding(4, 2, rng=np.random.default_rng(0))
        embedding.weight.data[:] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

        np.testing.assert_array_equal(embedding(2).data, [5.0, 6.0])
        np.testing.assert_array_equal(embedding([[3, 0], [1, 3]]).data,
                                      [[[7.0, 8.0], [1.0, 2.0]], [[3.0, 4.0], [7.0, 8.0]]])

    def test_gradients_accumulate_for_repeated_indices(self):
        embedding = Embedding(4, 2, rng=np.random.default_rng(0))

        (embedding([[0, 2], [2, 1]]) * Leaf([1.0, 10.0])).sum().wire()

        np.testing.assert_array_equal(embedding.weight.grad,
                                      [[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [0.0, 0.0]])

    def test_rejects_invalid_dimensions_and_indices(self):
        with self.assertRaises(ValueError):
            Embedding(0, 2, rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            Embedding(2, 0, rng=np.random.default_rng(0))

        embedding = Embedding(2, 3, rng=np.random.default_rng(0))
        with self.assertRaises(TypeError):
            embedding([0.0, 1.0])
        with self.assertRaises(ValueError):
            embedding([-1, 0])
        with self.assertRaises(ValueError):
            embedding([0, 2])

    def test_passes_table_dimensions_to_a_custom_initializer(self):
        rng = np.random.default_rng(7)

        def initializer(received_rng, num_embeddings, embedding_dim):
            self.assertIs(received_rng, rng)
            self.assertEqual((num_embeddings, embedding_dim), (2, 3))
            return np.full((num_embeddings, embedding_dim), 0.25)

        embedding = Embedding(2, 3, rng=rng, initializer=initializer)

        np.testing.assert_array_equal(embedding.weight.data, np.full((2, 3), 0.25))
