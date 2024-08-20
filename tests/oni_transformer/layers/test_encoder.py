import torch

import oni_transformer.layers as L


class TestEncoder:
    def test_forward(self):
        # Arrange
        token_vec_dim = 12
        num_heads = 3
        x = torch.randn(2, 3, token_vec_dim)
        encoder = L.Encoder(6, token_vec_dim, num_heads)

        # Act
        y = encoder(x)

        # Assert
        assert y.shape == (2, 3, token_vec_dim)
