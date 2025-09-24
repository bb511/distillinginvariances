# Graph convolution networks.

import numpy as np

import torch
import torch.nn as nn
import torch_geometric


class DGCNNPaper(nn.Module):
    """Dynamic graph convolutional neural network.

    This version has 4 layers. The default hyperparameters are identical to the ones
    used in the original paper Wang et.al. 2019 - Dynamic Graph CNN for learning on
    point clouds. For more details on how this network works, see the paper.

    Args:
        input_dim: The input dimension of the features.
        output_dim: The output dimension, equal to the number of classes in the dataset.
        output_mlp: The layers of the output mlp, after the edge conv layers.
        activ: The activation function to use between the layers of all the MLPs in
            this network, including the edge conv mlps.
        k: number of neighbours in the edge convolutional layers.
        aggreg: The aggregation function in the edge convolutional layers.
        dropout_rate: The dropout rate for the dropout applied to the layers of the
            output MLP, not the edge convolutional MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_mlp: list,
        linear_dim: int = 1024,
        k: int = 20,
        aggr: str = "max",
        activ: str = "relu",
        dropout_rate: float = 0.5,
    ):
        super(DGCNNPaper, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.aggr = aggr
        self.activ = activ

        self.conv1 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP(
                [self.input_dim] + [64],
                act=self.activ,
                act_kwargs={"negative_slope": 0.2},
                norm="batch_norm",
            ),
            self.k,
            self.aggr,
        )
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP(
                [64] + [64],
                act=self.activ,
                act_kwargs={"negative_slope": 0.2},
                norm="batch_norm",
            ),
            self.k,
            self.aggr,
        )
        self.conv3 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP(
                [64] + [128],
                act=self.activ,
                act_kwargs={"negative_slope": 0.2},
                norm="batch_norm",
            ),
            self.k,
            self.aggr,
        )
        self.conv4 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP(
                [128] + [256],
                act=self.activ,
                act_kwargs={"negative_slope": 0.2},
                norm="batch_norm",
            ),
            self.k,
            self.aggr,
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(64 + 64 + 128 + 256, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.mlp = torch_geometric.nn.MLP(
            [linear_dim] + output_mlp + [output_dim],
            dropout=dropout_rate,
            norm="batch_norm",
            act=self.activ,
            act_kwargs={"negative_slope": 0.2},
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        linear_out = self.linear_layer(torch.cat([x1, x2, x3, x4], dim=1))
        poolin_out = torch_geometric.nn.global_max_pool(linear_out, batch)
        classi_out = self.mlp(poolin_out)

        return classi_out

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data)


class DGCNNAlt(nn.Module):
    """Dynamic graph convolutional neural network.

    As above but with only 2 edge conv layers, that have deeper MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_mlp: list,
        linear_dim: int = 1024,
        k: int = 20,
        aggr: str = "max",
        activ: str = "relu",
        dropout_rate: float = 0.5,
    ):
        super(DGCNNAlt, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.aggr = aggr
        self.activ = activ

        self.conv1 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([self.input_dim] + [64, 64, 64], act=self.activ),
            self.k,
            self.aggr,
        )
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([64] + [128], act=self.activ), self.k, self.aggr
        )

        self.linear_layer = nn.Linear(64 + 128, linear_dim)

        self.mlp = torch_geometric.nn.MLP(
            [linear_dim] + output_mlp + [output_dim], dropout=dropout_rate, norm=None
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)

        linear_out = self.linear_layer(torch.cat([x1, x2], dim=1))
        poolin_out = torch_geometric.nn.global_max_pool(linear_out, batch)
        classi_out = self.mlp(poolin_out)

        return classi_out

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data)
