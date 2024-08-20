import torch

import oni_transformer.layers as L


class TestDecoder:
    def test_forward(self):
        # Arrange
        token_vec_dim = 12
        num_heads = 4
        decoder = L.Decoder(num_decoder_blocks=6, token_vec_dim=token_vec_dim, num_heads=num_heads)
        x = torch.randn(2, 4, token_vec_dim)
        memory = torch.randn(2, 4, token_vec_dim)

        # Act
        out = decoder(x, memory)

        # Assert
        assert out.shape == (2, 4, token_vec_dim)
