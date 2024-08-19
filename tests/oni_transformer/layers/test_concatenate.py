import torch

import oni_transformer.layers as L


class TestConcatenate:
    def test_forward(self):
        # Arrange
        x1 = torch.tensor([[1, 2], [3, 4]])
        x2 = torch.tensor([[5, 6], [7, 8]])
        layer = L.Concatenate()

        # Act
        y = layer.forward(x1, x2)

        # Assert
        assert y.shape == (2, 4)
        assert torch.equal(y, torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))
