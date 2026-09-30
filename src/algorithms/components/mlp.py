"""The student of every distillation: a plain MLP over the flattened sample."""

import torch
from torch import nn

from src.algorithms.components import FeatureNet


class MLPStudent(FeatureNet):
    """Fully connected ReLU network exposing each hidden activation.

    :param in_dim: width after flattening, ``constituents * features`` for the jets and
        ``pixels`` for MNIST.
    :param required_features: the columns the flattening assumes, set per dataset
        because the same class serves both.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (64, 32, 32, 32),
        output_dim: int = 5,
        required_features: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.required_features = list(required_features) if required_features else None
        sizes = [in_dim, *hidden]
        self.hidden = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden))]
        )
        self.output = nn.Linear(hidden[-1], output_dim)

    def forward_with_hidden(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the logits and the post-activation output of every hidden layer."""
        value = x.flatten(start_dim=1)
        activations = []
        for layer in self.hidden:
            value = torch.relu(layer(value))
            activations.append(value)
        return self.output(value), activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(x)[0]
