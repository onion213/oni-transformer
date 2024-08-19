import torch

import oni_transformer.layers as L


class TestSelfAttention:
    def test_forward(self):
        # Arrange
        x = torch.rand(2, 10, 24)
        layer = L.SelfAttention(24, num_heads=8)

        # Act
        y = layer(x)

        # Assert
        assert y.shape == (2, 10, 24)
