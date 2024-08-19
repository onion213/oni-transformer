import torch.nn as nn

from oni_transformer.layers.multi_head_attention import MultiHeadAttention


class SelfAttention(nn.Module):
    def __init__(self, token_vec_dim: int, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.token_vec_dim = token_vec_dim
        self.num_heads = num_heads
        self.affined_token_vec_dim = self.token_vec_dim // self.num_heads
        if self.affined_token_vec_dim * self.num_heads != self.token_vec_dim:
            raise ValueError("token_vec_dim must be divisible by num_heads")

        self.multi_head_attention = MultiHeadAttention(
            q_token_vec_dim=self.token_vec_dim,
            affined_qk_vec_dim=self.affined_token_vec_dim,
            affined_v_vec_dim=self.affined_token_vec_dim,
            num_heads=self.num_heads,
            output_token_vec_dim=self.token_vec_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x, mask=None):
        # validate input shape
        assert x.dim() == 3
        assert x.size(2) == self.token_vec_dim

        return self.multi_head_attention(x, x, x, mask=mask)
