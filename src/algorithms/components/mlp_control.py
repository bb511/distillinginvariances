"""The control teacher of the Deep Sets cell: an MLP over the flattened constituents.

A port of ``MLPClassifier`` of the sibling repository, jet_tagging_gdl
(src/algorithms/mlp.py and components/mlp.py). Its network is a Flatten followed by an
MLP holding the layers under ``MLP.net``, so its checkpoints load with ``strict=True``
once ``net.1.net.`` is renamed to ``net.layers.``.
"""

import torch
from torch import nn

from src.algorithms.components import FeatureNet


def _dense_stack(
    in_dim: int, widths: tuple[int, ...], out_dim: int, batchnorm: bool
) -> nn.Sequential:
    """Linear, BatchNorm1d (or Identity) and ReLU per hidden width, then a Linear.

    The Identity keeps the sibling's layer indices when the batch norm is off.
    """
    dims = [in_dim, *widths]
    layers: list[nn.Module] = []
    for width_in, width_out in zip(dims, widths):
        norm = nn.BatchNorm1d(width_out) if batchnorm else nn.Identity()
        layers += [nn.Linear(width_in, width_out), norm, nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-1], out_dim))


class MLPControl(FeatureNet):
    """Flatten (jets, constituents, 3) constituent-major and read it with an MLP.

    It has no symmetry of its own; the hint is its last hidden layer after the ReLU,
    which the output layer reads.

    :param in_dim: width after flattening, ``constituents * 3``.
    :param hidden: widths of the hidden layers, the last being the hint's.
    """

    required_features = ["pt", "etarel", "phirel"]

    def __init__(
        self,
        in_dim: int = 150,
        hidden: tuple[int, ...] = (128, 64, 32),
        output_dim: int = 5,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.hint_dim = hidden[-1]
        self.layers = _dense_stack(in_dim, tuple(hidden), output_dim, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(x)[0]

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the last hidden activation, (jets, ``hint_dim``)."""
        hint = self.layers[:-1](x.flatten(start_dim=1))
        return self.layers[-1](hint), hint
