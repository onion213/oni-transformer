import torch
import torch.nn as nn

from oni_transformer.layers.add_and_normalize import AddAndNormalize
from oni_transformer.layers.multi_head_attention import MultiHeadAttention
from oni_transformer.layers.position_wise_feed_forwarding import PositionWiseFeedForwarding
from oni_transformer.layers.self_attention import SelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, token_vec_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()

        self.token_vec_dim = token_vec_dim

        self.masked_self_attention = SelfAttention(token_vec_dim, num_heads, dropout_rate)
        self.add_norm1 = AddAndNormalize()
        if token_vec_dim % num_heads != 0:
            raise ValueError("token_vec_dim must be divisible by num_heads")
        self.multi_head = MultiHeadAttention(
            q_token_vec_dim=token_vec_dim,
            affined_qk_vec_dim=token_vec_dim // num_heads,
            affined_v_vec_dim=token_vec_dim // num_heads,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.add_norm2 = AddAndNormalize()
        self.feed_forward = PositionWiseFeedForwarding(token_vec_dim)
        self.add_norm3 = AddAndNormalize()

    def forward(self, outputs: torch.Tensor, encoded_inputs: torch.Tensor) -> torch.Tensor:
        # Assert input shape
        assert outputs.dim() == encoded_inputs.dim() == 3
        assert outputs.shape == encoded_inputs.shape
        assert outputs.size(2) == encoded_inputs.size(2) == self.token_vec_dim

        # mask for preventing attention to future tokens
        mask = torch.triu(torch.ones(outputs.size(1), outputs.size(1)), diagonal=1)

        x = self.add_norm1(outputs, self.masked_self_attention(outputs, mask=mask))
        x = self.add_norm2(x, self.multi_head(encoded_inputs, encoded_inputs, x))
        x = self.add_norm3(x, self.feed_forward(x))
        return x
