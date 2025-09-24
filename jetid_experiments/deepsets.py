# DeepSets implementation from DeepSets - Zaheer et. al.

import numpy as np

import torch
import torch.nn as nn


class PermutationEquivariantMean(nn.Module):
    """Permutation equivariant layer using the mean operation.

    Args:
        input_dim: The input dimension of this layer.
        output_dim: The output dimension of this layer.
    """

    def __init__(self, input_dim, output_dim):
        super(PermutationEquivariantMean, self).__init__()
        # Two linear layers, denoted gamma and lambda in the paper.
        self.Gamma = nn.Linear(input_dim, output_dim)
        self.Lambda = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x_mean = x.mean(1, keepdim=True)
        x_mean = self.Lambda(x_mean)
        x = self.Gamma(x)

        x = x - x_mean

        return x


class PermutationEquivariantMax(nn.Module):
    """Permutation equivariant layer using the maximum operation.

    Args:
        input_dim: The input dimension of this layer.
        output_dim: The output dimension of this layer.
    """

    def __init__(self, input_dim, output_dim):
        super(PermutationEquivariantMax, self).__init__()
        # Two linear layers, denoted gamma and lambda in the paper.
        self.Gamma = nn.Linear(input_dim, output_dim)
        self.Lambda = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x_max = x.max(1, keepdim=True)
        x_max = self.Lambda(x_mean)
        x = self.Gamma(x)

        x = x - x_max

        return x


def activ_string_to_torch(activ: str):
    """Converts activation name from string to a torch object."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(
            f"Activation {activation} not implemented! Go to deepsets.py and add it."
        )

    return activation


class DeepSetsEquivariant(nn.Module):
    """DeepSets permutation equivariant.

    For details on how this network works see Zaheer et. al. 2018 - DeepSets.
    Args:
        input_dim: The input dimension of the features.
        phi_layers: The layers of the phi MLP, applied on the input and before the
            aggregation operation.
        rho_layers: The layers of the rho MLP, applied after the aggregation operation.
        activ: The activation function to use between the layers of both phi and rho.
        aggreg: The aggregation operation performed over the 1st dimension (node level)
            of the data. This lies between the two MLPs.
        dropout: The dropout rate for the dropout applied to the layers of rho.
        output_dim: The output dimension, equal to the number of classes in the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        phi_layers: list,
        rho_layers: list,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
    ):
        super(DeepSetsEquivariant, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        self.phi_layers.insert(0, self.input_dim)
        for nlayer in range(len(self.phi_layers) - 1):
            layer = PermutationEquivariantMean(
                self.phi_layers[nlayer], self.phi_layers[nlayer + 1]
            )
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)

        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")

        self.rho_layers.insert(0, self.phi_layers[-1])
        self.rho_layers.append(self.output_dim)
        for nlayer in range(len(self.rho_layers) - 1):
            layer = nn.Linear(self.rho_layers[nlayer], self.rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(self.rho_layers) - 2:
                break
            rho.append(activ)

        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {
            "mean": lambda: torch.mean,
            "max": lambda: torch.max,
        }
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation

    def forward(self, x):
        x = data.pos
        agg_output = self.agg(phi_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)

        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)

        return output


class DeepSetsInvariant(nn.Module):
    """DeepSets permutation invariant.

    For details on how this network works see Zaheer et. al. 2018 - DeepSets.
    The difference between this network and its equivariant counterpart is that it does
    not use any of the permutation equivariant layers in the rho and phi networks.

    Args:
        input_dim: The input dimension of the features.
        phi_layers: The layers of the phi MLP, applied on the input and before the
            aggregation operation.
        rho_layers: The layers of the rho MLP, applied after the aggregation operation.
        activ: The activation function to use between the layers of both phi and rho.
        aggreg: The aggregation operation performed over the 1st dimension (node level)
            of the data. This lies between the two MLPs.
        dropout: The dropout rate for the dropout applied to the layers of rho.
        output_dim: The output dimension, equal to the number of classes in the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        phi_layers: list,
        rho_layers: str,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
    ):
        super(DeepSetsInvariant, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        self.phi_layers.insert(0, self.input_dim)
        for nlayer in range(len(self.phi_layers) - 1):
            layer = nn.Linear(self.phi_layers[nlayer], self.phi_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)

        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")

        self.rho_layers.insert(0, self.phi_layers[-1])
        self.rho_layers.append(self.output_dim)
        for nlayer in range(len(self.rho_layers) - 1):
            layer = nn.Linear(self.rho_layers[nlayer], self.rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(self.rho_layers) - 2:
                break
            rho.append(activ)
        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {"mean": lambda: torch.mean, "max": lambda: torch.max}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation

    def forward(self, x):
        phi_output = self.phi(x)
        agg_output = self.agg(phi_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)

        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)

        return output
