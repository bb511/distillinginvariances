"""LorentzNet, the Lorentz-invariant jets teacher of arXiv:2201.08187 (Gong et al.).

A port of ``LorentzNetGraph`` of the sibling repository, jet_tagging_gdl
(src/algorithms/lorentznet.py): the submodules keep its names and layer order, so its
checkpoints load with ``strict=True`` once the ``denorm_scales`` its Lightning module
holds is renamed into the network.

A jet is a fully connected graph over its constituents, self-loops excluded, with no
beam nodes. Each node carries its four-momentum :math:`(E, p_x, p_y, p_z)` and one
Lorentz scalar, the signed log of its squared mass. The sibling computes that mass from
the four-momentum the network is handed; this study reads it from the datamodule's
`~src.data.components.augmentation.MASS_COLUMN` instead (``node_scalar="stored"``), so a
transformed jet keeps the node scalars of its canonical row.

The metric is :math:`(+, -, -, -)` and the momenta are in GeV throughout: the edge
invariants are dimensionful and the signed log compresses them assuming that scale, so a
batch reaches `forward` through `prepare`, which de-normalises and reorders it.
"""

import torch
from torch import nn

from src.algorithms.components import FeatureNet
from src.data.components.augmentation import MASS_COLUMN, get_norm_scale

FOUR_MOMENTUM = ["px", "py", "pz", "e"]
NODE_SCALARS = ("computed", "stored")


def normsq4(p: torch.Tensor) -> torch.Tensor:
    """Minkowski square :math:`p \\cdot p` over the last axis, metric (+, -, -, -)."""
    psq = p.pow(2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Minkowski product :math:`p \\cdot q` over the last axis, metric (+, -, -, -)."""
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    """Signed log :math:`\\mathrm{sign}(p)\\,\\log(|p| + 1)`.

    This is equation (5) of arXiv:2201.08187. The edge invariants span several orders of
    magnitude in GeV squared; the signed log brings them into the linear layers' range
    while keeping the sign that separates timelike from spacelike.
    """
    return torch.sign(p) * torch.log(p.abs() + 1)


def _message_mlp(n_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * n_hidden + 2, n_hidden, bias=False),
        nn.BatchNorm1d(n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
    )


def _scalar_mlp(n_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * n_hidden + 1, n_hidden),
        nn.BatchNorm1d(n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
    )


def _coordinate_mlp(n_hidden: int) -> nn.Sequential:
    last = nn.Linear(n_hidden, 1, bias=False)
    nn.init.xavier_uniform_(last.weight, gain=0.001)
    return nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), last)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block: one round of message passing over a flat node
    set.

    Two edge attributes enter the messages, the norm of :math:`x_i - x_j` and the
    product :math:`x_i \\cdot x_j`.

    :param n_hidden: width of the node scalars and of the messages.
    :param c_weight: step size of the coordinate update.
    :param last_layer: drop the coordinate update, whose result no later block reads.
    """

    def __init__(
        self, n_hidden: int, c_weight: float = 1.0, last_layer: bool = False
    ) -> None:
        super().__init__()
        self.c_weight = c_weight
        self.phi_e = _message_mlp(n_hidden)
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.phi_h = _scalar_mlp(n_hidden)
        self.phi_x = None if last_layer else _coordinate_mlp(n_hidden)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated ``(h, x)``.

        :param h: node scalars, shape (nodes, n_hidden).
        :param x: node four-momenta, shape (nodes, 4).
        :param src: source node of each edge, shape (edges,).
        :param dst: destination node of each edge, shape (edges,).
        :param node_attr: the Lorentz scalar of each node, shape (nodes, 1).
        """
        x_diff = x[src] - x[dst]
        m = self._messages(h, x, x_diff, src, dst)
        if self.phi_x is not None:
            x = self._update_coordinates(x, x_diff, m, src)
        return self._update_scalars(h, m, node_attr, src), x

    def _messages(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """The (edges, n_hidden) messages, each gated by its own learned weight."""
        norms = psi(normsq4(x_diff)).unsqueeze(1)
        dots = psi(dotsq4(x[src], x[dst])).unsqueeze(1)
        m = self.phi_e(torch.cat([h[src], h[dst], norms, dots], dim=1))
        return m * self.phi_m(m)

    def _update_scalars(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        node_attr: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Add to each node the response to its summed incoming messages.

        Padded nodes receive no message but are updated too, as in the sibling; in
        training they enter the batch statistics of phi_h, in evaluation they reach
        neither a real node nor the pool.
        """
        agg = h.new_zeros(h.shape[0], m.shape[1]).index_add_(0, src, m)
        return h + self.phi_h(torch.cat([h, agg, node_attr], dim=1))

    def _update_coordinates(
        self, x: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Move each node along the mean of its weighted edge differences.

        The step is a combination of four-momentum differences, so it transforms like
        one and the block stays equivariant.
        """
        # tanh bounds the gate at 1, so no step exceeds |x_diff| and the next
        # block's signed log cannot overflow. Not the reference implementation's
        # per-component clamp at +-100 GeV: a rotation mixes the components, so the
        # clamp is not equivariant, and a trained gate exceeds it on most edges
        # (flip rates of 8 to 20 % under the probe, 2026-09-03).
        trans = x_diff * torch.tanh(self.phi_x(m))
        step = x.new_zeros(x.shape).index_add_(0, src, trans)
        count = x.new_zeros(x.shape).index_add_(0, src, torch.ones_like(trans))
        return x + self.c_weight * step / count.clamp(min=1)


def _blocks(n_hidden: int, n_layers: int, c_weight: float) -> nn.ModuleList:
    return nn.ModuleList(
        [
            LGEB(n_hidden, c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ]
    )


def _validated(node_scalar: str) -> str:
    if node_scalar not in NODE_SCALARS:
        raise ValueError(
            f"node_scalar must be one of {NODE_SCALARS}, got {node_scalar!r}."
        )
    return node_scalar


def _decoder(n_hidden: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n_hidden, output_dim),
    )


class LorentzNetGraph(FeatureNet):
    """LorentzNet over a batch of jets held as a fixed-width four-momentum tensor.

    The hint is the pooled node scalars the decoder reads, of width ``n_hidden``.

    :param n_hidden: width of the node scalars inside the blocks, and of the hint.
    :param c_weight: step size of the coordinate updates.
    :param dropout: inside the decoder, after its hidden layer.
    :param nconstituents: constituent slots per jet, the second axis of a batch.
    :param node_scalar: ``computed``, the sibling's float32 :math:`E^2 - |p|^2` of the
        four-momentum handed in, or ``stored``, the `MASS_COLUMN` of the batch, which a
        transform leaves as it was. The default is the sibling's, so a checkpoint whose
        hyperparameters predate the switch loads as it was imported; the state dict is
        the same for both.
    """

    def __init__(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        output_dim: int = 5,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        nconstituents: int = 50,
        node_scalar: str = "computed",
    ) -> None:
        super().__init__()
        self.node_scalar = _validated(node_scalar)
        self.n_hidden = self.hint_dim = n_hidden
        self.embedding = nn.Linear(1, n_hidden)
        self.blocks = _blocks(n_hidden, n_layers, c_weight)
        self.graph_dec = _decoder(n_hidden, output_dim, dropout)
        self._register_edges(nconstituents)
        self._scales_cached = False
        # The factors from the stored columns [px, py, pz, e] back to GeV.
        self.register_buffer("denorm_scales", torch.ones(4))

    @property
    def required_features(self) -> list[str]:
        return FOUR_MOMENTUM + ([MASS_COLUMN] if self.node_scalar == "stored" else [])

    def _register_edges(self, n_nodes: int) -> None:
        """The edges of one fully connected jet, self-loops excluded."""
        idx = torch.arange(n_nodes)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        self.register_buffer("edge_src", src[src != dst], persistent=False)
        self.register_buffer("edge_dst", dst[src != dst], persistent=False)

    def prepare(
        self, x: torch.Tensor, norm_scales: dict[str, float] | None = None
    ) -> torch.Tensor:
        """Return the batch as four-momenta in GeV, ordered [E, px, py, pz], followed by
        the squared mass in GeV^2 when the node scalar is stored.

        :param x: (jets, constituents, features), the columns [px, py, pz, e] in the
            datamodule's normalisation, then the unscaled `MASS_COLUMN` when the node
            scalar is stored.
        :param norm_scales: the datamodule's factors back to GeV, cached on the first
            call that supplies them; None keeps the cached factors.
        """
        if norm_scales is not None and not self._scales_cached:
            self._cache_scales(norm_scales)
        pmu = (x[..., :4] * self.denorm_scales.to(x.dtype))[..., [3, 0, 1, 2]]
        return torch.cat([pmu, x[..., 4:]], dim=-1)

    def _cache_scales(self, norm_scales: dict[str, float]) -> None:
        """Store the per-column factors to GeV of [px, py, pz, e]."""
        scales = [get_norm_scale(norm_scales, name) for name in FOUR_MOMENTUM]
        self.denorm_scales.copy_(torch.tensor(scales))
        self._scales_cached = True

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(pmu)[0]

    def forward_with_hint(self, pmu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the class logits and the pooled node scalars, which are the hint.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, padded
            constituents as all-zero rows; (jets, constituents, 5) with the squared mass
            in GeV^2 last when the node scalar is stored.
        """
        n_jets, n_slots, _ = pmu.shape
        real = pmu[..., 0] != 0.0
        src, dst = self._batch_edges(real)
        scalars = self._node_scalars(pmu).reshape(n_jets * n_slots, 1)
        h, x = self.embedding(scalars), pmu[..., :4].reshape(n_jets * n_slots, 4)
        for block in self.blocks:
            h, x = block(h, x, src, dst, scalars)
        pooled = self._pool(h, real)
        return self.graph_dec(pooled), pooled

    def _node_scalars(self, pmu: torch.Tensor) -> torch.Tensor:
        """Psi of each node's squared mass in GeV^2, shape (jets, constituents).

        A massless constituent, about half of the record's, has a computed m^2 at the
        rounding of its components and of E^2 - |p|^2, O(1e-7) E^2 in float32, which
        changes under any transformation; a pion's is 0.0195 GeV^2. The stored m^2 was
        computed once, in float64, from the canonical row, and no transform changes it.
        """
        if self.node_scalar == "stored":
            return psi(pmu[..., 4])
        return psi(normsq4(pmu[..., :4]))

    def _batch_edges(self, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index the edges that join two real constituents, over the flattened node
        axis; padded constituents are dropped to reduce the cost of the graphs.

        :param real: (jets, constituents) boolean, true on a real constituent.
        :return: two (edges,) index tensors into the flattened jets * constituents axis.
        """
        valid = real[:, self.edge_src] & real[:, self.edge_dst]
        jet_idx, edge_idx = valid.nonzero(as_tuple=True)
        offset = jet_idx * real.shape[1]
        return self.edge_src[edge_idx] + offset, self.edge_dst[edge_idx] + offset

    def _pool(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Average the node scalars of each jet over its real constituents.

        Over the real constituents rather than the slots, unlike the reference
        implementation, so that widening the padding leaves the logits alone.
        """
        counts = real.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        masked = h * real.reshape(-1, 1).to(h.dtype)
        return masked.view(real.shape[0], -1, self.n_hidden).sum(dim=1) / counts
