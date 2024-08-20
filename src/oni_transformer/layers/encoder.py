import torch.nn as nn

from oni_transformer.layers.encoder_block import EncoderBlock
from oni_transformer.layers.positional_encoder import PositionalEncoder


class Encoder(nn.Module):
    def __init__(self, num_encoder_blocks: int, token_vec_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.positional_encoder = PositionalEncoder(token_vec_dim)
        self.encoder_blocks = list(
            EncoderBlock(token_vec_dim, num_heads, dropout_rate) for _ in range(num_encoder_blocks)
        )

    def forward(self, x):
        x = self.positional_encoder(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
