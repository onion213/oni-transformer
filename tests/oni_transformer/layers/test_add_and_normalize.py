import numpy as np
import torch

import oni_transformer.layers as L


class TestAddAndNormalize:
    def test_forward(self):
        # Arrange
        x = torch.tensor([[[0, 2, 3], [4, 5, 6]]], dtype=torch.float32)
        y = torch.tensor([[[1, 0, -2], [10, 11, 12]]], dtype=torch.float32)
        layer = L.AddAndNormalize()
        expected = torch.tensor(
            [
                [
                    [1 / (2 + np.e), np.e / (2 + np.e), 1 / (2 + np.e)],
                    [
                        1 / (1 + np.e**2 + np.e**4),
                        np.e**2 / (1 + np.e**2 + np.e**4),
                        np.e**4 / (1 + np.e**2 + np.e**4),
                    ],
                ]
            ],
            dtype=torch.float32,
        )

        # Act
        result = layer(x, y)

        # Assert
        assert result.shape == expected.shape
        assert torch.allclose(result, expected)
