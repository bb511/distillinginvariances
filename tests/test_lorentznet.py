import pytest
import torch
from torch import nn

from src.algorithms.components.lorentznet import LorentzNetGraph, normsq4
from src.import_teacher import map_state_dict
from tests.conftest import jets

# The import's key map for a jet_tagging_gdl LorentzNet checkpoint.
SIBLING_RENAMES = {"denorm_scales": "net.denorm_scales"}


def _net(
    nconstituents: int = 8,
    dtype: torch.dtype = torch.float64,
    node_scalar: str = "computed",
) -> LorentzNetGraph:
    torch.manual_seed(3)
    net = LorentzNetGraph(
        c_weight=0.1, nconstituents=nconstituents, node_scalar=node_scalar
    )
    return net.to(dtype).eval()


def _sibling_state(h: int = 72, n_layers: int = 6) -> dict[str, torch.Tensor]:
    # The keys and shapes of the sibling's lorentznet/nconst50/baseline epoch_037.ckpt,
    # h being n_hidden.
    norm = {f"1.{k}": (h,) for k in ("weight", "bias", "running_mean", "running_var")}
    norm["1.num_batches_tracked"] = ()
    head = {"3.weight": (h, h), "3.bias": (h,)}
    mlps = {
        "phi_e": {"0.weight": (h, 2 * h + 2), **norm, **head},
        "phi_m": {"0.weight": (1, h), "0.bias": (1,)},
        "phi_h": {"0.weight": (h, 2 * h + 1), "0.bias": (h,), **norm, **head},
        "phi_x": {"0.weight": (h, h), "0.bias": (h,), "2.weight": (1, h)},
    }
    shapes = {"denorm_scales": (4,), "net.embedding.weight": (h, 1)}
    shapes["net.embedding.bias"] = (h,)
    for i in range(n_layers):
        for name, layers in mlps.items():
            if name != "phi_x" or i < n_layers - 1:
                shapes |= {f"net.blocks.{i}.{name}.{k}": s for k, s in layers.items()}
    decoder = {"0.weight": (h, h), "0.bias": (h,), "3.weight": (5, h), "3.bias": (5,)}
    shapes |= {f"net.graph_dec.{k}": s for k, s in decoder.items()}
    return {k: torch.rand(s) for k, s in shapes.items()}


def _lorentz_transformation(seed: int = 5) -> torch.Tensor:
    # A rotation, then a boost of rapidity 1.5 along a random axis, on [E, px, py, pz].
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[1:, 1:] = q * torch.linalg.det(q)
    axis = torch.randn(3, generator=generator, dtype=torch.float64)
    axis = axis / axis.norm()
    xi = torch.tensor(1.5, dtype=torch.float64)
    boost = torch.eye(4, dtype=torch.float64)
    boost[0, 0] = torch.cosh(xi)
    boost[0, 1:] = boost[1:, 0] = torch.sinh(xi) * axis
    boost[1:, 1:] += (torch.cosh(xi) - 1) * torch.outer(axis, axis)
    return boost @ rotation


def test_the_sibling_checkpoint_layout_loads_strictly() -> None:
    state = map_state_dict(_sibling_state(), [], SIBLING_RENAMES)
    holder = nn.Module()
    holder.net = LorentzNetGraph(nconstituents=50)

    holder.load_state_dict(state, strict=True)

    assert torch.equal(holder.net.denorm_scales, state["net.denorm_scales"])


@torch.no_grad()
def test_the_logits_of_a_massive_batch_are_lorentz_invariant_in_float64() -> None:
    net, transformation = _net(), _lorentz_transformation()
    pmu = net.prepare(jets(n=4).double())

    moved = pmu @ transformation.T

    assert (moved - pmu).abs().max() > 1.0
    assert torch.allclose(net(moved), net(pmu), rtol=0, atol=1e-10)


@torch.no_grad()
def test_the_padding_leaves_the_logits_alone() -> None:
    bare, narrow, wide, x = _net(5), _net(8), _net(12), jets(n=4).double()
    padded = torch.zeros(4, 12, 4, dtype=torch.float64)
    padded[:, 2:10] = x

    logits = narrow(narrow.prepare(x))

    assert torch.allclose(wide(wide.prepare(padded)), logits, rtol=0, atol=1e-12)
    assert torch.allclose(bare(bare.prepare(x[:, :5])), logits, rtol=0, atol=1e-12)


@torch.no_grad()
def test_the_hint_is_the_pooled_vector_the_decoder_reads() -> None:
    net = _net(dtype=torch.float32)
    logits, hint = net.forward_with_hint(net.prepare(jets(n=4)))

    assert hint.shape == (4, net.hint_dim) and net.hint_dim == 72
    assert torch.equal(net.graph_dec(hint), logits)


def _massless(n: int = 4) -> torch.Tensor:
    # (jets, slots, 5): [px, py, pz, e, m2] of massless constituents, padded slots zero.
    x = jets(n=n, columns=5)
    x[..., 3] = x[..., :3].norm(dim=-1)
    x[..., 4] = 0.0
    return x


@torch.no_grad()
def test_the_stored_scalar_fed_the_computed_mass_is_the_sibling_network() -> None:
    computed = _net(dtype=torch.float32)
    stored = _net(dtype=torch.float32, node_scalar="stored")
    x = jets(n=4)
    pmu = computed.prepare(x)
    with_mass = torch.cat([x, normsq4(pmu).unsqueeze(-1)], dim=-1)

    assert stored.required_features == ["px", "py", "pz", "e", "m2"]
    assert stored.state_dict().keys() == computed.state_dict().keys()
    assert torch.equal(stored.prepare(with_mass)[..., :4], pmu)
    assert torch.allclose(
        stored(stored.prepare(with_mass)), computed(pmu), rtol=0, atol=1e-6
    )
    with pytest.raises(ValueError, match="node_scalar"):
        LorentzNetGraph(node_scalar="massless")


@torch.no_grad()
def test_the_stored_scalar_leaves_a_transformed_massless_jet_invariant() -> None:
    net, transformation = _net(node_scalar="stored"), _lorentz_transformation()
    pmu = net.prepare(_massless().double())
    moved = torch.cat([pmu[..., :4] @ transformation.T, pmu[..., 4:]], dim=-1)

    assert (moved - pmu).abs().max() > 1.0
    assert torch.allclose(net(moved), net(pmu), rtol=0, atol=1e-10)


@torch.no_grad()
def test_a_float32_transform_moves_computed_node_scalars_not_stored() -> None:
    stored = _net(dtype=torch.float32, node_scalar="stored")
    computed = _net(dtype=torch.float32)
    pmu = stored.prepare(_massless())
    p4 = (pmu[..., :4].double() @ _lorentz_transformation().T).float()
    moved = torch.cat([p4, pmu[..., 4:]], dim=-1)

    assert torch.equal(stored._node_scalars(moved), stored._node_scalars(pmu))
    assert not torch.equal(computed._node_scalars(moved), computed._node_scalars(pmu))
