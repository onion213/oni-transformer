import torch

import oni_transformer.layers as L


class TestPositionalEncoding:
    def test_forward(self):
        # Arrange
        x = torch.zeros(1, 100, 20)
        layer = L.PositionalEncoder(20)

        # Act
        y = layer(x)

        # Assert
        assert y.shape == (1, 100, 20)
