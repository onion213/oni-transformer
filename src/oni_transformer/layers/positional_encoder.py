import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, token_vec_dim: int, max_len=512):
        super(PositionalEncoder, self).__init__()
        self.token_vec_dim = token_vec_dim
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        positional_encodings = self.get_positional_encodings(seq_len)
        return x + positional_encodings

    def get_positional_encodings(self, seq_len: int) -> torch.Tensor:
        positional_encodings = torch.zeros(self.max_len, self.token_vec_dim)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.token_vec_dim, 2) * -(torch.log(torch.tensor(10000.0)) / self.token_vec_dim)
        )
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        return positional_encodings[:seq_len].unsqueeze(0)
