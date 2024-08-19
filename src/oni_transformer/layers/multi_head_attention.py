import torch
import torch.nn as nn

from oni_transformer.layers.affine import Affine
from oni_transformer.layers.concatenate import Concatenate
from oni_transformer.layers.single_head_attention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        q_token_vec_dim: int,
        k_token_vec_dim: int | None = None,
        v_token_vec_dim: int | None = None,
        affined_qk_vec_dim: int | None = None,
        affined_v_vec_dim: int | None = None,
        num_heads: int = 8,
        output_token_vec_dim: int | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # deal with default values
        self.q_token_vec_dim = q_token_vec_dim
        self.k_token_vec_dim = k_token_vec_dim if k_token_vec_dim is not None else self.q_token_vec_dim
        self.v_token_vec_dim = v_token_vec_dim if v_token_vec_dim is not None else self.q_token_vec_dim
        self.affined_qk_vec_dim = affined_qk_vec_dim if affined_qk_vec_dim is not None else self.q_token_vec_dim
        self.affined_v_vec_dim = affined_v_vec_dim if affined_v_vec_dim is not None else self.v_token_vec_dim
        self.num_heads = num_heads
        self.output_token_vec_dim = (
            output_token_vec_dim if output_token_vec_dim is not None else self.affined_v_vec_dim * self.num_heads
        )
        self.dropout_rate = dropout_rate

        self.heads = list(
            SingleHeadAttention(
                self.q_token_vec_dim,
                self.k_token_vec_dim,
                self.v_token_vec_dim,
                self.affined_qk_vec_dim,
                self.affined_v_vec_dim,
                self.dropout_rate,
            )
            for _ in range(self.num_heads)
        )
        self.concat = Concatenate()
        self.output_affine = Affine(self.affined_v_vec_dim * self.num_heads, self.output_token_vec_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """Forward path.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, token_vector_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, token_vector_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, vector_dim).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, token_vector_dim).
        """
        # validate the input shapes
        assert q.dim() == k.dim() == v.dim() == 3  # q,k and v must be 3-d tensor
        assert q.size(0) == k.size(0) == v.size(0)  # batch size must be the same
        assert k.size(1) == v.size(1)  # seq_len of k and v must be the same
        assert q.size(2) == self.q_token_vec_dim
        assert k.size(2) == self.k_token_vec_dim
        assert v.size(2) == self.v_token_vec_dim

        # calculate multi-head attention
        head_outputs = [head(q, k, v, mask) for head in self.heads]
        concatenated_heads = self.concat(*head_outputs)
        output = self.output_affine(concatenated_heads)
        return output
