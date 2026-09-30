"""The networks of the study, and the interface the Lightning modules expect of them."""

import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


def activation_module(name: str) -> nn.Module:
    """Return the activation layer named by *name*.

    :raises ValueError: when the name is none of `ACTIVATIONS`.
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got {name!r}")
    return ACTIVATIONS[name]()


def mlp_head(
    in_dim: int,
    hidden: tuple[int, ...],
    out_dim: int,
    activation: str,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Linear stack ``in_dim -> hidden -> out_dim``, with no activation on the output.

    Dropout, where asked for, sits before every linear layer, as in the old DeepSets.
    """
    sizes = [in_dim, *hidden, out_dim]
    head = nn.Sequential()
    for index in range(len(sizes) - 1):
        if 0.0 < dropout < 1.0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            head.append(activation_module(activation))
    return head


class FeatureNet(nn.Module):
    """A network that states which columns it reads and how wide its hint is.

    :cvar required_features: the columns the network needs, in its own order; None asks
        for the batch as it arrives.
    :cvar hint_dim: width of the hint this network hands a student, 0 for a network that
        teaches nobody.
    """

    required_features: list[str] | None = None
    hint_dim: int = 0

    def prepare(
        self, x: torch.Tensor, norm_scales: dict[str, float] | None = None
    ) -> torch.Tensor:
        """Last step before the forward pass, once the columns have been selected.

        The identity here; LorentzNet de-normalises and reorders its four-momenta.
        """
        return x

    def forward_with_hint(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits beside the layer a student is trained to reproduce."""
        raise NotImplementedError(f"{type(self).__name__} serves no hint.")
