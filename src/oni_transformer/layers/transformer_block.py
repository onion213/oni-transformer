import torch
import torch.nn as nn

from oni_transformer.layers.add_and_normalize import AddAndNormalize
from oni_transformer.layers.position_wise_feed_forwarding import PositionWiseFeedForwarding
from oni_transformer.layers.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, token_vec_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = SelfAttention(token_vec_dim, num_heads)
        self.attention_norm = AddAndNormalize()
        self.ffn = PositionWiseFeedForwarding(token_vec_dim)
        self.ffn_norm = AddAndNormalize()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attention_norm(x, self.attention(x, mask))
        x = self.ffn_norm(x, self.ffn(x))
        return x
