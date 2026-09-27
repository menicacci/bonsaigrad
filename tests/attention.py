import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Sequential
from bonsaigrad.nn.attention import SelfAttention
from bonsaigrad.optim import SGD


class TestSelfAttention(unittest.TestCase):

    def test_causal_mask_excludes_future_tokens(self):
        attention = SelfAttention(2, head_dim=2)
        for projection in (attention.Q, attention.K, attention.V):
            projection.weight.data[:] = np.eye(2)
            projection.bias.data[:] = 0
        inputs = [[1.0, 0.0], [0.0, 1.0]]

        output = attention(inputs).data

        np.testing.assert_allclose(output[0], inputs[0])
        weights = np.exp([0.0, 1.0 / np.sqrt(2)])
        np.testing.assert_allclose(output[1], weights / weights.sum())
        unmasked = SelfAttention(2, head_dim=2, causal=False)
        for source, target in zip((attention.Q, attention.K, attention.V),
                                  (unmasked.Q, unmasked.K, unmasked.V)):
            target.weight.data[:] = source.weight.data
            target.bias.data[:] = source.bias.data
        self.assertGreater(unmasked(inputs).data[0, 1], 0.0)

    def test_parameters_train_through_sequential(self):
        attention = SelfAttention(4, head_dim=2)
        model = Sequential(attention)
        parameters = model.parameters()
        self.assertEqual(len(parameters), 6)
        self.assertTrue(all(isinstance(parameter, Leaf) for parameter in parameters))
        optimizer = SGD(parameters, learning_rate=0.1)
        model(Leaf(np.ones((2, 3, 4)))).sum().wire()
        self.assertTrue(any(np.any(parameter.grad != 0) for parameter in parameters))
        optimizer.step()

    def test_small_embedding_and_invalid_dimensions(self):
        self.assertEqual(SelfAttention(3).head_dim, 1)
        for kwargs in ({"embedding_dim": 0}, {"embedding_dim": 4, "head_dim": 0}):
            with self.assertRaises(ValueError):
                SelfAttention(**kwargs)
