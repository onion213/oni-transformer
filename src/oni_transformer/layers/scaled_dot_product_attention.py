import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Given query, key, and value, this modele calculates as follows:
    1. Calculate the dot product of query and key.
    2. Scale the result by dividing the square root of the dimension of key.
    3. Apply a softmax function to obtain the weights on the values.
    4. Multiply the weights with the values to get the output.
    """

    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward path.

        Calculate scaled dot-product attention in the following steps:
        1. Calculate the dot product of query and key.
            product = q * k^try:
            resulting shape: (batch_size, seq_len, attention_seq_len)
        2. Scale the result by dividing the square root of the dimension of key.
            scaled_product = product / sqrt(token_vec_dim)
            resulting shape: (batch_size, seq_len, attention_seq_len)
        3. Apply a softmax function to obtain the weights on the values.
            attention = softmax(scaled_product)
            resulting shape: (batch_size, seq_len, attention_seq_len)
        4. Multiply the weights with the values to get the output.
            output = attention * v
            resulting shape: (batch_size, seq_len, output_token_vec_dim)

        Args:
            q (torch.Tensor): Query tensor with shape (batch_size, seq_len, token_vec_dim).
            k (torch.Tensor): Key tensor with shape (batch_size, attention_seq_len, token_vec_dim).
            v (torch.Tensor): Value tensor with shape (batch_size, attention_seq_len, output_token_vec_dim).
            mask (torch.Tensor): Mask tensor with shape (batch_size, seq_len, attention_seq_len).
        Returns:
            output (torch.Tensor): Output tensor with shape (batch_size, seq_len, output_token_vec_dim).
        """
        # validate the input shapes
        assert q.dim() == k.dim() == v.dim() == 3  # q,k,v must be 3D tensors
        assert q.size(0) == k.size(0) == v.size(0)  # the first dimension of q,k,v must be the same
        assert q.size(2) == k.size(2)  # the last dimension of q and k must be the same
        assert k.size(1) == v.size(1)  # the second dimension of k and v must be the same

        # calculate the dot product attention
        token_vec_dim = q.size(-1)
        scaled_product = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(token_vec_dim)
        if mask is not None:
            scaled_product = scaled_product.masked_fill(mask == 0, -1e9)
        softmax = nn.Softmax(dim=-1)
        attention = self.dropout(softmax(scaled_product))
        output = torch.matmul(attention, v)
        return output
