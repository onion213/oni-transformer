import numpy as np
import torch

import oni_transformer.layers as L


class TestSingleHeadAttention:
    def test_forward(self):
        # Arrange
        q = torch.Tensor([[[2, 1, 2]]])
        k = torch.Tensor([[[2, 2, 1], [2, 1, 2], [1, 2, 2]]])
        v = torch.Tensor([[[2, 2, 1], [2, 1, 2], [1, 2, 2]]])
        layer = L.SingleHeadAttention(q_token_vec_dim=3, dropout_rate=0.0)
        layer.q_affine.net.weight = torch.nn.Parameter(torch.Tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))
        layer.q_affine.net.bias = torch.nn.Parameter(torch.Tensor([-6, -6, -6]))
        layer.k_affine.net.weight = torch.nn.Parameter(torch.Tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))
        layer.k_affine.net.bias = torch.nn.Parameter(torch.Tensor([-6, -6, -6]))
        layer.v_affine.net.weight = torch.nn.Parameter(torch.Tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))
        layer.v_affine.net.bias = torch.nn.Parameter(torch.Tensor([-6, -6, -6]))
        a = np.exp(1 / np.sqrt(3))
        expected = torch.tensor([[[(1 + a), 2, (1 + a)]]], dtype=torch.float32) / (2 + a)

        # Act
        output = layer(q, k, v)

        # Assert
        assert output.shape == (1, 1, 3)
        assert torch.allclose(output, expected)
