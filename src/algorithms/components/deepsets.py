"""Permutation-invariant Deep Sets (arXiv:1703.06114), the jets teacher of the cheap
regime.

A port of ``DeepSetsNet`` of the sibling repository, jet_tagging_gdl
(src/algorithms/deepsets.py), without its jet moments: ``phi`` and ``rho`` keep its
submodule names and layer order, so its checkpoints load with ``strict=True``.
"""

import torch

from src.algorithms.components import FeatureNet, activation_module, mlp_head

AGGREGATIONS = ("sum", "max", "sum_over_n")


def _masked_pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`; ``sum_over_n`` divides the masked sum by
        the slot count, the same for every jet, where a mean would divide by each jet's
        multiplicity.
    """
    if aggregation == "max":
        return h.masked_fill(~real, float("-inf")).amax(dim=1)
    total = (h * real).sum(dim=1)
    return total if aggregation == "sum" else total / h.shape[1]


class DeepSetsNet(FeatureNet):
    """Apply phi to every constituent, pool over the real ones, then read the pool with
    rho.

    A padded constituent is an all-zero row and is left out of the pool, so the output
    ignores the padding and is invariant under any permutation of the slots. The hint is
    the pooled vector rho reads.

    :param phi_layers: widths of the per-constituent network, the last being the width
        of the pooled vector and of the hint.
    :param rho_layers: hidden widths of the head applied to the pooled vector.
    :param aggregation: one of `AGGREGATIONS`.
    :param dropout: before every linear layer of rho, as in the sibling.
    """

    required_features = ["pt", "etarel", "phirel"]

    def __init__(
        self,
        in_dim: int = 3,
        phi_layers: tuple[int, ...] = (64, 64, 32),
        rho_layers: tuple[int, ...] = (64, 32),
        output_dim: int = 5,
        activation: str = "relu",
        aggregation: str = "sum",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}"
            )
        self.aggregation = aggregation
        self.hint_dim = phi_layers[-1]
        # The sibling's phi ends in an activation and its rho does not; both keep their
        # linear layers at the even indices.
        self.phi = mlp_head(in_dim, phi_layers[:-1], phi_layers[-1], activation)
        self.phi.append(activation_module(activation))
        self.rho = mlp_head(phi_layers[-1], rho_layers, output_dim, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(x)[0]

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the pooled phi output, (jets, ``hint_dim``)."""
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        pooled = _masked_pool(self.phi(x), real, self.aggregation)
        return self.rho(pooled), pooled
