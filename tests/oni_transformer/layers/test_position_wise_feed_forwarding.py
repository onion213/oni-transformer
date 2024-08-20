import torch

import oni_transformer.layers as L


class TestPositionWiseFeedForwarding:
    def test_forward(self):
        # Arrange
        layer = L.PositionWiseFeedForwarding(8)
        x = torch.randn(2, 4, 8)

        # Act
        out = layer(x)

        # Assert
        assert out.shape == (2, 4, 8)
