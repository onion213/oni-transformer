import torch
from torch import nn


class Affine(nn.Module):
    """Affine Transformation Layer"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)
