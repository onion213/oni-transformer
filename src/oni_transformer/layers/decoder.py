import torch.nn as nn

from oni_transformer.layers.affine import Affine
from oni_transformer.layers.decoder_block import DecoderBlock
from oni_transformer.layers.positional_encoder import PositionalEncoder


class Decoder(nn.Module):
    def __init__(self, num_decoder_blocks: int, token_vec_dim: int, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.positional_encoder = PositionalEncoder(token_vec_dim)
        self.decoder_blocks = list(
            DecoderBlock(token_vec_dim, num_heads, dropout_rate) for _ in range(num_decoder_blocks)
        )
        self.affine = Affine(token_vec_dim, token_vec_dim)

    def forward(self, outputs, encoded_inputs):
        x = self.positional_encoder(outputs)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoded_inputs)
        x = self.affine(x)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        return x
