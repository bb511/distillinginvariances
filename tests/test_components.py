import torch

from src.algorithms.components.cnn import CNNInvariant
from src.algorithms.components.deepsets import DeepSetsInvariant
from src.algorithms.components.lorentznet import LorentzNetGraph
from src.algorithms.components.mlp import MLPStudent
from tests.conftest import jets


def test_the_cnn_serves_its_pooled_hint() -> None:
    logits, hint = CNNInvariant().forward_with_hint(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 10) and hint.shape == (4, 64)


def test_deepsets_is_permutation_invariant_and_hints_the_aggregated_phi() -> None:
    net, x = DeepSetsInvariant().eval(), torch.randn(4, 8, 3)
    logits, hint = net.forward_with_hint(x)

    assert torch.allclose(logits, net(x[:, torch.randperm(8)]), atol=1e-5)
    assert torch.allclose(hint, net.phi(x).mean(dim=1))


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
    assert logits.shape == (6, 5) and hint.shape == (6, 96)
