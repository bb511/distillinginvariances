# MLP model classes.

import numpy as np

import torch
import torch.nn as nn


def activ_string_to_torch(activ: str):
    """Converts activation name from string to a torch objects."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "leaky_relu": lambda: nn.LeakyReLU(),
    }
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(
            f"Activation {activation} not implemented! Go to deepsets.py and add it."
        )

    return activation


class MLPBasic(nn.Module):
    """Basic MLP with variable number of layers.

    Args:
        input_dim: The input dimension of the flattenet data.
        layers: List for which the length is the depth of the MLP and each element
            specifies the width of each layer.
        output_dim: The output dimension of the network, equal to the number of classes
            in the data.
        activ: String specifying the activation between each layer of the network.
    """

    def __init__(self, input_dim: int, layers: list, output_dim: int, activ: str):
        super(MLPBasic, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.layers = layers
        self.output_dim = output_dim

        self._construct_mlp()

    def _construct_mlp(self):
        """Construct the MLP using a sequential module in pytorch."""
        self.layers = [self.input_dim] + self.layers + [self.output_dim]

        self.mlp = nn.Sequential()
        for nlayer in range(len(self.layers) - 1):
            self.mlp.add_module(
                f"linear_{nlayer}",
                nn.Linear(self.layers[nlayer], self.layers[nlayer + 1]),
            )

            if nlayer < len(self.layers) - 2:
                activation = activ_string_to_torch(self.activ)
                self.mlp.add_module(f"activation_{nlayer}", activation)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)


class MLPReged(nn.Module):
    """Same as the basic MLP above but with dropout applied to each layer."""

    def __init__(
        self,
        input_dim: int,
        layers: list,
        output_dim: int,
        activ: str,
        dropout_rate: float = 0.5,
    ):
        super(MLPReged, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.layers = layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self._construct_mlp()

    def _construct_mlp(self):
        """Construct the MLP using a sequential module in pytorch."""
        self.layers = [self.input_dim] + self.layers + [self.output_dim]

        self.mlp = nn.Sequential()
        for nlayer in range(len(self.layers) - 1):
            self.mlp.add_module(
                f"linear_{nlayer}",
                nn.Linear(self.layers[nlayer], self.layers[nlayer + 1]),
            )

            if nlayer < len(self.layers) - 2:
                activation = activ_string_to_torch(self.activ)
                self.mlp.add_module(f"activation_{nlayer}", activation)
                self.mlp.add_module(
                    f"dropout_{nlayer}", nn.Dropout(p=self.dropout_rate)
                )
                self.mlp.add_module("flatten", nn.Flatten(start_dim=1))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)
