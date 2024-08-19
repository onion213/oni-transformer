import numpy as np
import torch

import oni_transformer.layers as L


class TestScaledDotProductAttention:
    def test_forward(self):
        """Test the forward method of the ScaledDotProductAttention layer.
        k = [1, 0, 1]
        q = v = [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ]

        k*q^T = [1, 2, 1] (dot product)
        k*q^T/sqrt(3) = [1/sqrt(3), 2/sqrt(3), 1/sqrt(3)] (scaled dot product)
        softmax(k*q^T/sqrt(3)) = [1, a, 1]/(2+a) (where a = exp(1/sqrt(3)))
        softmax(k*q^T/sqrt(3))*v = [1+a, 2, 1+a]/(2+a) (=out)
        """
        # Arrange
        q = torch.tensor([[[1, 0, 1]]], dtype=torch.float32)
        k = torch.tensor([[[1, 1, 0], [1, 0, 1], [0, 1, 1]]], dtype=torch.float32)
        v = torch.tensor([[[1, 1, 0], [1, 0, 1], [0, 1, 1]]], dtype=torch.float32)
        l = L.ScaledDotProductAttention(dropout_rate=0)
        a = np.exp(1 / np.sqrt(3))
        expected = torch.tensor([[[(1 + a), 2, (1 + a)]]], dtype=torch.float32) / (2 + a)

        # Act
        out = l(q, k, v)

        # Assert
        assert out.shape == (1, 1, 3)
        assert torch.allclose(out, expected)
