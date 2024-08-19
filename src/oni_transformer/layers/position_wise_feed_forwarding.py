import torch
import torch.nn as nn

from oni_transformer.layers.affine import Affine


class PositionWiseFeedForwarding(nn.Module):
    def __init__(self, vec_dim: int):
        super().__init__()
        self.affine1 = Affine(vec_dim, vec_dim)
        self.activation = nn.ReLU()
        self.affine2 = Affine(vec_dim, vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, seq_len, vec_dim)

        Returns:
            torch.Tensor: (batch_size, seq_len, vec_dim)
        """
        return self.affine2(self.activation(self.affine1(x)))
