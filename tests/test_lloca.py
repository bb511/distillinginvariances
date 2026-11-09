from functools import partial

import pytest
import torch
from lloca.backbone.transformer import Transformer
from lloca.equivectors.mlp import MLPVectors
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from torch import nn

from src.algorithms.classifier import Classifier
from src.algorithms.components.lloca import FRAMES, ORTHO_KWARGS, LLoCaClassifier
from src.data.components.augmentation import apply_group
from src.import_teacher import map_state_dict
from tests.conftest import FOUR_MOMENTUM, jets

ADAM = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}
SMALL = {
    "num_blocks": 2,
    "num_heads": 2,
    "attn_scalars": 2,
    "attn_vectors": 1,
    "mlp_factor": 2,
    "framesnet_hidden": 8,
}


def _sibling_state(frames: str, dropout: float) -> dict[str, torch.Tensor]:
    # The layout of a jet_tagging_gdl LLoCa checkpoint, built from the package classes
    # as its LLoCaClassifier builds them: the network under net., the module-level
    # denorm_scales, and the float64 edge standardisation of the first training batch.
    sibling = nn.Module()
    equivectors = partial(
        MLPVectors,
        num_scalars=0,
        hidden_channels=8,
        num_layers_mlp=2,
        dropout_prob=None,
    )
    sibling.framesnet = FRAMES[frames](
        equivectors=equivectors, mass_reg=2.5e-4, ortho_kwargs=ORTHO_KWARGS[frames]
    )
    sibling.canonicalise = TensorRepsTransform(TensorReps("1x1n"))
    sibling.net = Transformer(
        in_channels=4,
        attn_reps="2x0n+1x1n",
        out_channels=5,
        num_blocks=2,
        num_heads=2,
        mlp_factor=2,
        dropout_prob=dropout or None,
    )
    state = {f"net.{k}": v for k, v in sibling.state_dict().items()}
    block = "net.framesnet.equivectors.block"
    state[f"{block}.edge_inited"] = torch.tensor(True)
    state[f"{block}.edge_mean"] = torch.tensor(-5.7, dtype=torch.float64)
    state[f"{block}.edge_std"] = torch.tensor(2.1, dtype=torch.float64)
    return state | {"denorm_scales": torch.tensor([1478.4, 1340.4, 3066.0, 3241.2])}


def _readout(t: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (t * real[..., None]).sum(dim=1) / real.sum(dim=1, keepdim=True)


def _initialised(net: LLoCaClassifier) -> LLoCaClassifier:
    # One training-mode pass sets the edge standardisation, as the first training batch
    # does.
    net.train()
    net(net.prepare(jets(seed=3)))
    return net.eval()


def _package_forward(
    net: LLoCaClassifier, pmu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # The sibling's forward, through the package's own Transformer.forward, and the
    # masked mean of what its output projection reads.
    real = pmu[..., 0] != 0.0
    p = pmu / net.units
    frames = net._frames(p, real)
    read = []
    hook = net.net.linear_out.register_forward_pre_hook(
        lambda module, args: read.append(args[0])
    )
    out = net.net(
        net.canonicalise(p, frames),
        frames,
        p_ref=(p * real[..., None]).sum(dim=1),
        attn_mask=real[:, None, None, :],
    )
    hook.remove()
    return _readout(out, real), _readout(read[0], real)


@pytest.mark.parametrize(
    "frames, dropout", [("z", 0.0), ("so3", 0.0), ("lorentz", 0.1)]
)
def test_a_sibling_checkpoint_loads_strictly(frames: str, dropout: float) -> None:
    state = _sibling_state(frames, dropout)
    net = {
        "_target_": "src.algorithms.components.lloca.LLoCaClassifier",
        "frames": frames,
        "dropout": dropout,
        **SMALL,
    }
    module = Classifier(net=net, num_classes=5, input_shape=(8, 4), optimizer=ADAM)

    module.load_state_dict(map_state_dict(state, ["denorm_scales"], {}), strict=True)

    loaded = module.state_dict()
    assert all(
        torch.equal(loaded[k], v.to(loaded[k].dtype))
        for k, v in state.items()
        if k != "denorm_scales"
    )


def test_the_hint_is_what_the_head_reads_and_the_logits_are_the_siblings() -> None:
    torch.manual_seed(0)
    net = _initialised(LLoCaClassifier(frames="z", **SMALL))
    pmu = net.prepare(jets())

    logits, hint = net.forward_with_hint(pmu)
    reference_logits, reference_hint = _package_forward(net, pmu)

    assert torch.equal(net(pmu), logits) and hint.shape == (len(pmu), net.hint_dim)
    assert torch.allclose(logits, reference_logits, atol=1e-6)
    assert torch.allclose(hint, reference_hint, atol=1e-6)


@pytest.mark.parametrize(
    "frames, exact, broken", [("z", "z", "so3"), ("so3", "so3", "lorentz")]
)
def test_the_logits_are_invariant_under_the_group_of_the_frames_alone(
    frames: str, exact: str, broken: str
) -> None:
    torch.manual_seed(0)
    net = _initialised(LLoCaClassifier(frames=frames, **SMALL)).double()
    batch = jets().double()

    logits = net(net.prepare(batch))
    moved = net(net.prepare(apply_group(exact, batch, FOUR_MOMENTUM)[0]))
    off = net(net.prepare(apply_group(broken, batch, FOUR_MOMENTUM)[0]))

    assert torch.allclose(moved, logits, atol=1e-9)
    assert (off - logits).abs().max() > 1e-6


def test_the_logits_ignore_the_padding_and_the_slot_order() -> None:
    torch.manual_seed(0)
    net = _initialised(LLoCaClassifier(frames="lorentz", **SMALL)).double()
    batch = jets().double()
    wider = torch.cat([batch, torch.zeros(6, 4, 4, dtype=batch.dtype)], dim=1)

    logits = net(net.prepare(batch))

    assert torch.allclose(net(net.prepare(wider)), logits, atol=1e-10)
    assert torch.allclose(
        net(net.prepare(wider[:, torch.randperm(12)])), logits, atol=1e-10
    )
