import torch
from torch import nn

from oni_transformer.layers.affine import Affine
from oni_transformer.layers.scaled_dot_product_attention import ScaledDotProductAttention


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        q_token_vec_dim: int,
        k_token_vec_dim: int | None = None,
        v_token_vec_dim: int | None = None,
        affined_qk_vec_dim: int | None = None,
        affined_v_vec_dim: int | None = None,
        dropout_rate: float = 0.1,
    ):
        """Initialize SingleHeadAttention.
        Args:
            token_vec_dim (int): token vector dimension
            output_token_vec_dim (int): output token vector dimension
        """
        super().__init__()

        # deal with default values
        if k_token_vec_dim is None:
            k_token_vec_dim = q_token_vec_dim
        if v_token_vec_dim is None:
            v_token_vec_dim = q_token_vec_dim
        if affined_qk_vec_dim is None:
            affined_qk_vec_dim = q_token_vec_dim
        if affined_v_vec_dim is None:
            affined_v_vec_dim = v_token_vec_dim

        # create layers
        self.q_token_vec_dim = q_token_vec_dim
        self.k_token_vec_dim = k_token_vec_dim
        self.v_token_vec_dim = v_token_vec_dim
        self.affined_qk_vec_dim = affined_qk_vec_dim
        self.affined_v_vec_dim = affined_v_vec_dim
        self.q_affine = Affine(q_token_vec_dim, affined_qk_vec_dim)
        self.k_affine = Affine(k_token_vec_dim, affined_qk_vec_dim)
        self.v_affine = Affine(v_token_vec_dim, affined_v_vec_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """forward path.

        Caluculate in the following steps.
        1. Affine transformation for query, key, and value.
        2. Scale dot product attention.

        Args:
            q (torch.Tensor): query tensor of shape (batch_size, seq_len, token_vec_dim)
            k (torch.Tensor): key tensor of shape (batch_size, attention_seq_len, token_vec_dim)
            v (torch.Tensor): value tensor of shape (batch_size, attention_seq_len, output_token_vec_dim)
        """
        # validate the input shapes
        assert q.dim() == k.dim() == v.dim() == 3  # q,k and v must be 3-d tensor
        assert q.size(0) == k.size(0) == v.size(0)  # batch size must be the same
        assert q.size(2) == self.q_token_vec_dim
        assert k.size(2) == self.k_token_vec_dim
        assert v.size(2) == self.v_token_vec_dim

        # affine transformation
        affined_q = self.q_affine(q)
        affined_k = self.k_affine(k)
        affined_v = self.v_affine(v)

        # scale dot product attention
        output = self.scaled_dot_product_attention(affined_q, affined_k, affined_v, mask)

        return output
