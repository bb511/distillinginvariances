import torch

from src.algorithms.components.lorentznet import LorentzNetGraph
from src.data.components.augmentation import apply_random_lorentz
from tests.conftest import FOUR_MOMENTUM, jets

# The logits of an untrained net are O(0.1); float32 moves them by less under a boost.
INVARIANCE_TOL = 2e-3


def _net(nconstituents: int = 8) -> LorentzNetGraph:
    torch.manual_seed(3)
    return LorentzNetGraph(nconstituents=nconstituents).eval()


@torch.no_grad()
def test_the_logits_are_invariant_only_with_co_transformed_beams() -> None:
    net, x = _net(), jets(n=4)
    boosted, matrices = apply_random_lorentz(x, FOUR_MOMENTUM, 1.0, None, True)

    rest = net(net.prepare(x))

    assert (
        rest - net(net.prepare(boosted), lorentz_matrix=matrices)
    ).abs().max() < INVARIANCE_TOL
    assert (rest - net(net.prepare(boosted))).abs().max() > INVARIANCE_TOL


@torch.no_grad()
def test_the_padding_width_leaves_the_logits_alone_and_the_hint_feeds_the_decoder() -> (
    None
):
    narrow, wide, x = _net(8), _net(12), jets(n=4)
    wide.load_state_dict(narrow.state_dict())
    padded = torch.zeros(4, 12, 4)
    padded[:, :8] = x

    logits, hint = narrow.forward_with_hint(narrow.prepare(x))

    assert torch.allclose(logits, wide(wide.prepare(padded)), atol=1e-6)
    assert torch.allclose(narrow.decode(hint), logits)
