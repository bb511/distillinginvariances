"""Translation-invariant CNN, the MNIST teacher."""

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms.components import FeatureNet, activation_module, mlp_head


class CNNInvariant(FeatureNet):
    """Convolutions with circular padding, pooled to a per-image vector, then a head.

    With circular padding a circular translation of the image translates every feature
    map, and the global average pooling then leaves one vector per image unchanged.

    :param phi_channels: channels of the convolutions, one entry per depth.
    :param hint_depth: depth, one-based, whose pooled output a student reproduces.
    """

    def __init__(
        self,
        input_channels: int = 1,
        phi_channels: tuple[int, ...] = (32, 64, 64),
        rho_layers: tuple[int, ...] = (128,),
        output_dim: int = 10,
        activation: str = "relu",
        kernel_size: int = 3,
        padding_mode: str = "circular",
        hint_depth: int = 3,
    ) -> None:
        super().__init__()
        self.hint_depth = hint_depth
        self.hint_dim = phi_channels[hint_depth - 1]
        self.phi = self._build_phi(
            input_channels, phi_channels, kernel_size, padding_mode, activation
        )
        self.rho = mlp_head(phi_channels[-1], rho_layers, output_dim, activation)

    @staticmethod
    def _build_phi(
        input_channels: int,
        channels: tuple[int, ...],
        kernel_size: int,
        padding_mode: str,
        activation: str,
    ) -> nn.Sequential:
        """One convolution and one activation per depth, the spatial size preserved."""
        sizes = [input_channels, *channels]
        phi = nn.Sequential()
        for index in range(len(sizes) - 1):
            phi.append(
                nn.Conv2d(
                    sizes[index],
                    sizes[index + 1],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                )
            )
            phi.append(activation_module(activation))
        return phi

    @staticmethod
    def _pool(feature_maps: torch.Tensor) -> torch.Tensor:
        """Average each channel of a (images, channels, h, w) map over the image."""
        return F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(self._pool(self.phi(x)))

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the pooled activations at `hint_depth`."""
        # Two modules per depth, so the activation of depth d sits at index 2d - 1.
        hint_at = 2 * self.hint_depth - 1
        hint = None
        for index, layer in enumerate(self.phi):
            x = layer(x)
            if index == hint_at:
                hint = self._pool(x)
        return self.rho(self._pool(x)), hint
