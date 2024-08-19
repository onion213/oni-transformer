import torch
from torch import nn

import oni_transformer.layers as L


class TestAffine:
    def test_forward(self):
        # Arrange
        x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        layer = L.Affine(3, 4)
        layer.net.weight = nn.Parameter(torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
        layer.net.bias = nn.Parameter(torch.Tensor([1, 2, 3, 4]))

        # Act
        y = layer(x)

        # Assert
        assert y.shape == (2, 4)
        assert torch.allclose(y, torch.Tensor([[15, 34, 53, 72], [33, 79, 125, 171]]))

    def test_backward(self):
        # Arrange
        x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        layer = L.Affine(3, 4)
        layer.net.weight = nn.Parameter(torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
        layer.net.bias = nn.Parameter(torch.Tensor([1, 2, 3, 4]))
        y = layer(x)
        dy = torch.ones_like(y)

        # Act
        y.backward(dy)

        # Assert
        assert torch.allclose(layer.net.weight.grad, torch.Tensor([[5, 7, 9], [5, 7, 9], [5, 7, 9], [5, 7, 9]]))
        assert torch.allclose(layer.net.bias.grad, torch.Tensor([2, 2, 2, 2]))
