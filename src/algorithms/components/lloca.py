"""LLoCa (arXiv:2505.20280), Lorentz local canonicalisation: a jets teacher whose logits
are invariant under the group its frames co-transform with.

A port of ``LLoCaClassifier`` of the sibling repository, jet_tagging_gdl
(src/algorithms/lloca.py). The frames-net, the frames and the LLoCa-Transformer come
from the ``lloca`` package (2.0.0, the sibling's version); the submodule names are the
sibling's, so its checkpoints load with ``strict=True``.

The metric is (+, -, -, -), the energy first, and the network reads GeV: `prepare`
returns the stored columns to GeV and reorders them, as the sibling's FourMomentumModule
did. The group is set by ``frames``: the full Lorentz group for ``lorentz``, rotations
about the beam times boosts along it for ``z``, rotations about the beam for ``so2`` and
SO(3) for ``so3``.
"""

from functools import partial

import torch
from lloca.backbone.transformer import Transformer
from lloca.equivectors.mlp import MLPVectors
from lloca.framesnet.equi_frames import (
    LearnedPDFrames,
    LearnedSO2Frames,
    LearnedSO3Frames,
    LearnedZFrames,
)
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from torch import nn

from src.algorithms.components import FeatureNet
from src.data.components.augmentation import get_norm_scale

FRAMES = {
    "lorentz": LearnedPDFrames,
    "z": LearnedZFrames,
    "so2": LearnedSO2Frames,
    "so3": LearnedSO3Frames,
}
# The sibling's settings, from the LLoCa tagging guide: Gram-Schmidt in float64, with
# eps_norm raised to 1e-10 for the rotation-only frames.
ORTHO_KWARGS = {
    key: {
        "use_float64": True,
        "method": "gramschmidt",
        "eps_norm": 1e-10 if key in ("so2", "so3") else 1e-15,
        "eps_reg": 1e-16,
        "eps_reg_lightlike": 1e-16,
    }
    for key in FRAMES
}


def _packed_ptr(real: torch.Tensor) -> torch.Tensor:
    """Offsets ``(jets + 1,)`` of each jet's constituents in the packed ``p[real]``."""
    ptr = real.new_zeros(real.shape[0] + 1, dtype=torch.long)
    ptr[1:] = real.sum(dim=1).cumsum(dim=0)
    return ptr


def _masked_mean(t: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Mean of (jets, constituents, width) over the real constituents, the sibling's
    readout term for term."""
    return (t * real[..., None]).sum(dim=1) / real.sum(dim=1, keepdim=True)


def _framesnet(
    frames: str, framesnet_hidden: int, framesnet_dropout: float, mass_reg: float
) -> nn.Module:
    # 0 becomes None: the package adds dropout layers to its MLP for any probability
    # but None, and they renumber the later linear layers in the state dict.
    equivectors = partial(
        MLPVectors,
        num_scalars=0,
        hidden_channels=framesnet_hidden,
        num_layers_mlp=2,
        dropout_prob=framesnet_dropout or None,
    )
    return FRAMES[frames](
        equivectors=equivectors, mass_reg=mass_reg, ortho_kwargs=ORTHO_KWARGS[frames]
    )


class LLoCaClassifier(FeatureNet):
    """LLoCa-Transformer over a fixed-width batch of jet four-momenta.

    Each real constituent is read in a frame of its own, predicted equivariantly from
    the whole jet, and the logits are the mean of the backbone's outputs over the real
    constituents. They are invariant under any permutation of the slots, under the group
    of ``frames``, and under the padding. The hint is what the head reads: the same mean
    of the backbone's last hidden output, before its output projection.

    :param frames: a key of `FRAMES`.
    :param attn_scalars: scalar channels of each attention head.
    :param attn_vectors: four-vector channels of each attention head.
    :param framesnet_hidden: width of the two hidden layers of the frames-net MLP.
    :param mass_reg: mass floor in ``units``, applied inside the frames-net alone; it
        keeps the pair sums the frames-net normalises off the light cone.
    :param units: GeV per unit of everything the package sees.
    :raises ValueError: when ``frames`` is not a key of `FRAMES`.
    """

    required_features = ["px", "py", "pz", "e"]

    def __init__(
        self,
        frames: str = "lorentz",
        num_blocks: int = 8,
        num_heads: int = 8,
        attn_scalars: int = 8,
        attn_vectors: int = 2,
        mlp_factor: int = 2,
        dropout: float = 0.0,
        framesnet_hidden: int = 32,
        framesnet_dropout: float = 0.0,
        mass_reg: float = 2.5e-4,
        units: float = 20.0,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        if frames not in FRAMES:
            raise ValueError(f"frames must be one of {list(FRAMES)}, got {frames!r}.")
        self.units = units
        self.framesnet = _framesnet(
            frames, framesnet_hidden, framesnet_dropout, mass_reg
        )
        self.canonicalise = TensorRepsTransform(TensorReps("1x1n"))
        self.net = Transformer(
            in_channels=4,
            attn_reps=f"{attn_scalars}x0n+{attn_vectors}x1n",
            out_channels=output_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_factor=mlp_factor,
            dropout_prob=dropout or None,
        )
        self.hint_dim = self.net.hidden_channels

    def prepare(
        self, x: torch.Tensor, norm_scales: dict[str, float] | None = None
    ) -> torch.Tensor:
        """Return the batch as four-momenta [E, px, py, pz] in GeV.

        :param x: (jets, constituents, 4), the stored columns [px, py, pz, e].
        :param norm_scales: the datamodule's factors back to GeV; None leaves the batch
            unscaled.
        """
        scales = [get_norm_scale(norm_scales, name) for name in self.required_features]
        return (x * x.new_tensor(scales))[..., [3, 0, 1, 2]]

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(pmu)[0]

    def forward_with_hint(self, pmu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the hint, (jets, ``output_dim``) and (jets,
        ``hint_dim``).

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        """
        real = pmu[..., 0] != 0.0
        p = pmu / self.units
        frames = self._frames(p, real)
        hidden = self._hidden(self.canonicalise(p, frames), frames, p, real)
        logits = _masked_mean(self.net.linear_out(hidden), real)
        return logits, _masked_mean(hidden, real)

    def _hidden(
        self, local: torch.Tensor, frames: Frames, p: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        """The backbone's last hidden output, (jets, constituents, ``hint_dim``).

        `Transformer.forward` of lloca 2.0.0 up to its output projection, with the jet
        momentum as the reference of ``preserve_variance`` and a key-padding mask. The
        padded query rows attend to the real keys and give finite values, which the
        masked mean never reads.
        """
        p_ref = (p * real[..., None]).sum(dim=1)
        self.net.attention.prepare_frames(frames, p_ref=p_ref)
        h = self.net.linear_in(local)
        for block in self.net.blocks:
            h = block(h, attn_mask=real[:, None, None, :])
        return h

    def _frames(self, p: torch.Tensor, real: torch.Tensor) -> Frames:
        """Predict the frame of every real constituent; a padded slot gets the identity.

        :param p: (jets, constituents, 4) four-momenta [E, px, py, pz] in ``units``.
        :param real: (jets, constituents), True on the real constituents.
        :return: frames of shape (jets, constituents, 4, 4).
        """
        # The edge convolution has no padding mask in the dense layout, hence the packed
        # one. float64, as the references feed it: in float32 the mass floor is below
        # the rounding of |p|^2 above about 14 GeV, and the massless pair masses fall to
        # zero, where the frames-net's sqrt has no finite gradient. The zero-width
        # scalars keep the parameters' dtype, so the frames-net's MLP does too.
        packed, ptr = p[real].double(), _packed_ptr(real)
        scalars = packed.new_zeros(packed.shape[0], 0, dtype=p.dtype)
        equivectors = self.framesnet.equivectors
        # The edge standardisation runs once, on the first training batch, into buffers
        # the checkpoint carries.
        if self.training and not equivectors.block.edge_inited:
            equivectors.init_standardization(packed, ptr)
        matrices = torch.eye(4).to(p).repeat(*real.shape, 1, 1)
        frames = self.framesnet(packed, scalars=scalars, ptr=ptr)
        matrices[real] = frames.matrices.to(p.dtype)
        return Frames(matrices)
