import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Forward path.

        Receives a list of tensors and concatenates them along the last dimension.
        All tensors must have the same shape except for the last dimension.

        Args:
            *xs (torch.Tensor): List of tensors.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        # validate input
        for x in xs[1:]:
            assert x.dim() == xs[0].dim(), "All tensors must have the same number of dimensions"
            for i in range(x.dim() - 1):
                assert x.size(i) == xs[0].size(i), "All tensors must have the same shape except for the last dimension"

        return torch.cat(xs, dim=-1)
