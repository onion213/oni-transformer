import torch

import oni_transformer.layers as L


class TestMultiHeadAttention:
    def test_forward(self):
        # Arrange
        q = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
        k = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
        v = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
        multi_head_attention = L.MultiHeadAttention(3, num_heads=4)

        # Act
        y = multi_head_attention(q, k, v)

        # Assert
        assert y.shape == (1, 2, 12)
