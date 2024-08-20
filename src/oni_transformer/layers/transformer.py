import torch.nn as nn

from oni_transformer.layers.decoder import Decoder
from oni_transformer.layers.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoders: int = 6,
        num_decoders: int = 6,
        token_vec_dim: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_encoders, token_vec_dim, num_encoder_heads, dropout_rate)
        self.decoder = Decoder(num_decoders, token_vec_dim, num_decoder_heads, dropout_rate)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        return self.decoder(tgt, memory)
