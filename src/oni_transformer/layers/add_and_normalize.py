import torch
import torch.nn as nn


class AddAndNormalize(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # validate input shapes
        assert x.dim() == y.dim() == 3, f"Input tensors must be 3D, got {x.dim()=} and {y.dim()=} respectively"
        assert x.shape == y.shape, f"Shapes of x and y must be same, got {x.shape=} and {y.shape=} respectively"

        return nn.Softmax(dim=-1)(x + y)
