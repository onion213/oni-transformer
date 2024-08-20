import torch

import oni_transformer.layers as L


class TestEncoderBlock:
    def test_forward(self):
        # Arrange
        x = torch.rand(2, 10, 24)
        layer = L.EncoderBlock(token_vec_dim=24, num_heads=8)

        # Act
        out = layer(x)

        # Assert
        assert out.shape == x.shape
