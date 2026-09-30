"""Permutation-invariant Deep Sets, the jets teacher of the cheap regime."""

from collections.abc import Callable

import torch
from torch import nn

from src.algorithms.components import FeatureNet, activation_module, mlp_head

AGGREGATIONS = {
    "mean": lambda t: t.mean(dim=1),
    "max": lambda t: t.max(dim=1).values,
}


class DeepSetsInvariant(FeatureNet):
    """Per-constituent phi, an aggregation over the constituents, then rho.

    :param phi_layers: widths of the per-constituent stack, one entry per depth.
    :param hint_depth: depth, one-based, whose aggregated output a student reproduces.
    """

    required_features = ["pt", "etarel", "phirel"]

    def __init__(
        self,
        in_dim: int = 3,
        phi_layers: tuple[int, ...] = (32, 32, 32),
        rho_layers: tuple[int, ...] = (32,),
        output_dim: int = 5,
        activation: str = "relu",
        aggregation: str = "mean",
        dropout: float = 0.0,
        hint_depth: int = 3,
    ) -> None:
        super().__init__()
        self.hint_depth = hint_depth
        self.hint_dim = phi_layers[hint_depth - 1]
        self.phi = self._build_phi(in_dim, phi_layers, activation)
        self.aggregate = self._get_aggregation(aggregation)
        self.rho = mlp_head(phi_layers[-1], rho_layers, output_dim, activation, dropout)

    @staticmethod
    def _build_phi(
        in_dim: int, layers: tuple[int, ...], activation: str
    ) -> nn.Sequential:
        """One linear layer and one activation per depth, applied constituent-wise.

        A padded constituent is an all-zero row, which the biases map to a non-zero
        value; it is not masked, as in the architecture of the paper.
        """
        sizes = [in_dim, *layers]
        phi = nn.Sequential()
        for index in range(len(sizes) - 1):
            phi.append(nn.Linear(sizes[index], sizes[index + 1]))
            phi.append(activation_module(activation))
        return phi

    @staticmethod
    def _get_aggregation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if name not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {list(AGGREGATIONS)}, got {name!r}"
            )
        return AGGREGATIONS[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(self.aggregate(self.phi(x)))

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the aggregated activations at `hint_depth`."""
        # Two modules per depth, so the activation of depth d sits at index 2d - 1.
        hint_at = 2 * self.hint_depth - 1
        hint = None
        for index, layer in enumerate(self.phi):
            x = layer(x)
            if index == hint_at:
                hint = self.aggregate(x)
        return self.rho(self.aggregate(x)), hint
