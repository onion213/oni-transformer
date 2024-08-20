import torch

import oni_transformer.layers as L


class TestDecoderBlock:
    def test_forward(self):
        # Arrange
        token_vec_dim = 12
        num_heads = 4
        encoded_inputs = torch.randn(2, 3, 12)
        outputs = torch.randn(2, 3, 12)
        layer = L.DecoderBlock(token_vec_dim, num_heads)

        # Act
        y = layer(encoded_inputs, outputs)

        # Assert
        assert y.shape == (2, 3, 12)
