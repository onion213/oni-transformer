import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, num_encoders: int = 6, num_decoders: int = 6, token_vec_dim: int = 512):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_encoders, token_vec_dim)
        self.decoder = Decoder(num_decoders, token_vec_dim)
        self.linear = nn.Linear(token_vec_dim, 1)
