"""Translation-invariant CNN, the MNIST teacher."""

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms.components import FeatureNet, activation_module, mlp_head


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
    padding = kernel_size // 2
    for c_in, c_out in zip(sizes[:-1], sizes[1:]):
        phi.append(
            nn.Conv2d(
                c_in, c_out, kernel_size, padding=padding, padding_mode=padding_mode
            )
        )
        phi.append(activation_module(activation))
    return phi


def _pool(feature_maps: torch.Tensor) -> torch.Tensor:
    """Average each channel of a (images, channels, h, w) map over the image."""
    return F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)


class CNNInvariant(FeatureNet):
    """Convolutions with circular padding, pooled to a per-image vector, then a head.

    With circular padding a circular translation of the image translates every feature
    map, and the global average pooling then leaves one vector per image unchanged. The
    hint is that pooled vector, the one rho reads.

    :param phi_channels: channels of the convolutions, one entry per depth, the last
        being the width of the pooled vector and of the hint.
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
    ) -> None:
        super().__init__()
        self.hint_dim = phi_channels[-1]
        self.phi = _build_phi(
            input_channels, phi_channels, kernel_size, padding_mode, activation
        )
        self.rho = mlp_head(phi_channels[-1], rho_layers, output_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(x)[0]

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the pooled output of the last convolution layer,
        (images, ``hint_dim``)."""
        pooled = _pool(self.phi(x))
        return self.rho(pooled), pooled
