import pytest
import torch
from torch import nn

from src.algorithms.components.cnn import CNNInvariant
from src.algorithms.components.deepsets import DeepSetsNet
from src.algorithms.components.lorentznet import LorentzNetGraph
from src.algorithms.components.mlp import MLPStudent
from tests.conftest import jets


def _sibling_dense(dims: list[int], final_activation: bool) -> nn.Sequential:
    # jet_tagging_gdl's components/mlp.py::_dense without dropout.
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if final_activation or i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _padded_constituents(real: int = 5, slots: int = 8) -> torch.Tensor:
    x = torch.randn(4, slots, 3, generator=torch.Generator().manual_seed(3))
    x[:, real:] = 0.0
    return x


@pytest.mark.parametrize("channels", [(16, 32), (32, 64, 64, 128)])
def test_the_cnn_hints_the_pooled_last_convolution_that_rho_reads(
    channels: tuple[int, ...],
) -> None:
    net, x = CNNInvariant(phi_channels=channels).eval(), torch.randn(4, 1, 28, 28)
    logits, hint = net.forward_with_hint(x)

    assert sum(isinstance(m, nn.Conv2d) for m in net.phi) == len(channels)
    assert net.hint_dim == channels[-1] and hint.shape == (4, channels[-1])
    assert torch.allclose(hint, net.phi(x).mean(dim=(2, 3)))
    assert torch.allclose(logits, net.rho(hint)) and logits.shape == (4, 10)
    assert torch.equal(net(x), logits)


def test_deepsets_loads_the_sibling_state_dict_strictly() -> None:
    sibling = nn.Module()
    sibling.phi = _sibling_dense([3, 64, 64, 32], final_activation=True)
    sibling.rho = _sibling_dense([32, 64, 32, 5], final_activation=False)
    net, x = DeepSetsNet().eval(), _padded_constituents()

    net.load_state_dict(sibling.state_dict(), strict=True)
    real = x.abs().sum(dim=-1, keepdim=True) > 0

    assert torch.allclose(net(x), sibling.rho((sibling.phi(x) * real).sum(dim=1)))


@pytest.mark.parametrize("aggregation", ["sum", "max"])
def test_deepsets_ignores_the_padding_and_the_order_of_the_constituents(
    aggregation: str,
) -> None:
    net, x = DeepSetsNet(aggregation=aggregation).eval(), _padded_constituents()
    shuffled = x[:, torch.randperm(8, generator=torch.Generator().manual_seed(4))]

    assert torch.allclose(net(x), net(x[:, :5]), atol=1e-5)
    assert torch.allclose(net(x), net(shuffled), atol=1e-5)


def test_deepsets_hints_the_pooled_phi_output_that_rho_reads() -> None:
    net, x = DeepSetsNet().eval(), _padded_constituents()
    logits, hint = net.forward_with_hint(x)

    assert hint.shape == (4, net.hint_dim) and net.hint_dim == 32
    assert torch.allclose(hint, net.phi(x[:, :5]).sum(dim=1), atol=1e-5)
    assert torch.allclose(logits, net.rho(hint))


def test_the_mlp_exposes_every_hidden_activation() -> None:
    net = MLPStudent(in_dim=96, hidden=(64, 32, 32, 32))
    logits, hidden = net.forward_with_hidden(torch.randn(4, 8, 12))

    assert logits.shape == (4, 5) and [h.shape[-1] for h in hidden] == [64, 32, 32, 32]
    assert torch.allclose(hidden[1], torch.relu(net.hidden[1](hidden[0])))


def test_lorentznet_prepare_scales_and_reorders_to_e_first() -> None:
    net, x = LorentzNetGraph(nconstituents=8).eval(), jets()
    scales = {"j1_px": 2.0, "j1_py": 3.0, "j1_pz": 4.0, "j1_e": 5.0}

    pmu = net.prepare(x, scales)
    logits, hint = net.forward_with_hint(pmu)

    assert torch.allclose(
        pmu, x[..., [3, 0, 1, 2]] * torch.tensor([5.0, 2.0, 3.0, 4.0])
    )
    assert torch.allclose(net.prepare(x), pmu)  # the factors are cached
    assert logits.shape == (6, 5) and hint.shape == (6, 72)
