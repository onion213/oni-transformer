import torch

import oni_transformer.layers as L


class TestTransformer:
    def test_forward(self):
        # Arrange
        token_vec_dim = 512
        src = torch.rand(2, 3, token_vec_dim)
        tgt = torch.rand(2, 3, token_vec_dim)
        transformer = L.Transformer()
        # Act
        output = transformer(src, tgt)

        # Assert
        assert output.shape == (2, 3, token_vec_dim)
