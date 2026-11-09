import torch
from torch import nn

from src.algorithms.components.mlp_control import MLPControl
from src.import_teacher import map_state_dict

# The import's key map for a jet_tagging_gdl MLP checkpoint.
SIBLING_RENAMES = {"net.1.net.": "net.layers."}


class _SiblingMLP(nn.Module):
    # jet_tagging_gdl's components/mlp.py::MLP at [128, 64, 32] with batch norm.
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for width_in, width_out in zip([150, 128, 64], [128, 64, 32]):
            layers += [nn.Linear(width_in, width_out), nn.BatchNorm1d(width_out)]
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers, nn.Linear(32, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sibling() -> nn.Sequential:
    torch.manual_seed(2)
    sibling = nn.Sequential(nn.Flatten(1), _SiblingMLP())
    for norm in (m for m in sibling.modules() if isinstance(m, nn.BatchNorm1d)):
        norm.running_mean.uniform_(-1.0, 1.0)
        norm.running_var.uniform_(0.5, 2.0)
    return sibling.eval()


def _imported(sibling: nn.Module) -> MLPControl:
    state = {f"net.{k}": v for k, v in sibling.state_dict().items()}
    holder = nn.Module()
    holder.net = MLPControl()
    holder.load_state_dict(map_state_dict(state, [], SIBLING_RENAMES), strict=True)
    return holder.net.eval()


@torch.no_grad()
def test_the_sibling_checkpoint_loads_strictly_and_flattens_in_its_order() -> None:
    sibling = _sibling()
    net, x = _imported(sibling), torch.randn(4, 50, 3)

    assert torch.equal(net(x), sibling(x))
    assert not torch.allclose(net(x.transpose(1, 2)), sibling(x), atol=1e-3)


@torch.no_grad()
def test_the_hint_is_the_last_hidden_activation_the_output_layer_reads() -> None:
    net = _imported(_sibling())
    logits, hint = net.forward_with_hint(torch.randn(4, 50, 3))

    assert hint.shape == (4, net.hint_dim) and net.hint_dim == 32
    assert (hint >= 0).all() and torch.equal(net.layers[-1](hint), logits)
