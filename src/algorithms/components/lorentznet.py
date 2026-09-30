"""LorentzNet with beam nodes, the jets teacher of arXiv:2201.08187 (Gong et al., 2022).

A jet is a fully connected graph over its constituents and the two beam nodes, self-
loops excluded. Each node carries its four-momentum :math:`(E, p_x, p_y, p_z)` in GeV
and two Lorentz scalars, the signed log of its mass in the channel that separates a beam
from a constituent.

The metric is :math:`(+, -, -, -)` and the momenta are in GeV throughout: the edge
invariants are dimensionful and the signed log compresses them assuming that scale, so a
batch reaches `forward` through `prepare`, which de-normalises and reorders it.
"""

import math

import torch
from torch import nn

from src.algorithms.components import FeatureNet
from src.data.components.augmentation import get_norm_scale


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


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block: one round of message passing over a flat node
    set.

    Two edge attributes enter the messages, the norm of :math:`x_i - x_j` and the
    product :math:`x_i \\cdot x_j`.

    :param n_scalar: node scalars per node, 2 with the beams and 1 without.
    :param c_weight: step size of the coordinate update.
    :param last_layer: drop the coordinate update, whose result no later block reads.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_scalar: int,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.c_weight = c_weight
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phi_m = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_scalar, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi_x = None
        if not last_layer:
            last = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.xavier_uniform_(last.weight, gain=0.001)
            self.phi_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), last
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_attr: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated ``(h, x)``.

        :param h: node scalars, shape (nodes, hidden_dim).
        :param x: node four-momenta, shape (nodes, 4).
        :param src: source node of each edge, shape (edges,).
        :param dst: destination node of each edge, shape (edges,).
        :param node_attr: the Lorentz scalars, shape (nodes, n_scalar).
        :param node_mask: true on a real node, shape (nodes,).
        """
        x_diff = x[src] - x[dst]
        m = self._messages(h, x, x_diff, src, dst)
        if self.phi_x is not None:
            x = self._update_coordinates(x, x_diff, m, src)
        return self._update_scalars(h, m, node_attr, src, node_mask), x

    def _messages(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """The (edges, hidden_dim) messages, each gated by its own learned weight."""
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
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add to each real node the response to its summed incoming messages.

        The padded nodes are left out, and so out of the batch statistics of phi_h,
        which is what keeps the logits independent of how many slots the padding fills.
        """
        agg = h.new_zeros(h.shape[0], m.shape[1]).index_add_(0, src, m)
        update = h.new_zeros(h.shape)
        update[node_mask] = self.phi_h(torch.cat([h, agg, node_attr], dim=1)[node_mask])
        return h + update

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


class LorentzNetGraph(FeatureNet):
    """LorentzNet over a batch of jets held as a fixed-width four-momentum tensor.

    :param hidden_dim: width of the node scalars inside the blocks, and of the hint.
    :param c_weight: step size of the coordinate updates.
    :param add_beams: prepend the two beam nodes of the reference implementation.
    :param beam_mass: mass of a beam node, in GeV.
    :param nconstituents: constituent slots per jet, the second axis of a batch.
    """

    required_features = ["px", "py", "pz", "e"]

    def __init__(
        self,
        hidden_dim: int = 96,
        n_layers: int = 4,
        output_dim: int = 5,
        c_weight: float = 1e-3,
        dropout: float = 0.2,
        add_beams: bool = True,
        beam_mass: float = 1.0,
        nconstituents: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hint_dim = hidden_dim
        self.add_beams = add_beams
        self.beam_mass = beam_mass
        self.n_scalar = 2 if add_beams else 1
        self.embedding = nn.Linear(self.n_scalar, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LGEB(
                    hidden_dim,
                    self.n_scalar,
                    c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._register_edges(nconstituents + (2 if add_beams else 0))
        self._scales_cached = False
        # The factors from the stored columns [px, py, pz, e] back to GeV.
        self.register_buffer("denorm_scales", torch.ones(4))

    def _register_edges(self, n_nodes: int) -> None:
        """The edges of one fully connected jet, self-loops excluded."""
        idx = torch.arange(n_nodes)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        self.register_buffer("edge_src", src[src != dst], persistent=False)
        self.register_buffer("edge_dst", dst[src != dst], persistent=False)

    def prepare(
        self, x: torch.Tensor, norm_scales: dict[str, float] | None = None
    ) -> torch.Tensor:
        """Return the batch as four-momenta in GeV, ordered [E, px, py, pz].

        :param x: (jets, constituents, 4) holding the record's columns [px, py, pz, e]
            in the datamodule's normalisation.
        :param norm_scales: the datamodule's factors back to GeV, cached on the first
            call that supplies them; None leaves the batch unscaled.
        """
        if norm_scales is not None and not self._scales_cached:
            self._cache_scales(norm_scales)
        return (x * self.denorm_scales.to(x.dtype))[..., [3, 0, 1, 2]]

    def _cache_scales(self, norm_scales: dict[str, float]) -> None:
        """Store the per-column factors to GeV, in the order of `required_features`."""
        scales = [get_norm_scale(norm_scales, name) for name in self.required_features]
        self.denorm_scales.copy_(torch.tensor(scales))
        self._scales_cached = True

    def forward(
        self, pmu: torch.Tensor, lorentz_matrix: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_with_hint(pmu, lorentz_matrix)[0]

    def forward_with_hint(
        self, pmu: torch.Tensor, lorentz_matrix: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the class logits and the pooled node scalars, which are the hint.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, padded
            constituents as all-zero rows.
        :param lorentz_matrix: (jets, 4, 4) on [E, px, py, pz], the transformation the
            batch has already been through. The beams are co-transformed by it, so they
            remain the same two physical nodes in every frame.
        """
        x, scalars, node_mask = self._nodes(pmu, lorentz_matrix)
        h = self._message_passing(x, scalars, node_mask)
        pooled = self._pool(h, node_mask)
        return self.decode(pooled), pooled

    def _nodes(
        self, pmu: torch.Tensor, lorentz_matrix: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (jets, nodes, 4) momenta, (jets, nodes, n_scalar) scalars and the
        (jets, nodes) real-node mask, the two beams first where they are added."""
        x = self._with_beams(pmu, lorentz_matrix) if self.add_beams else pmu
        return x, self._scalars(x), x[..., 0] != 0.0

    def _with_beams(
        self, pmu: torch.Tensor, lorentz_matrix: torch.Tensor | None
    ) -> torch.Tensor:
        """Prepend the beams (sqrt(1 + m^2), 0, 0, +-1) with m the beam mass in GeV."""
        energy = math.sqrt(1.0 + self.beam_mass**2)
        beams = pmu.new_tensor(
            [[[energy, 0.0, 0.0, 1.0], [energy, 0.0, 0.0, -1.0]]]
        ).expand(pmu.shape[0], 2, 4)
        if lorentz_matrix is not None:
            beams = beams @ lorentz_matrix.to(pmu.dtype).transpose(-1, -2)
        return torch.cat([beams, pmu], dim=1)

    def _scalars(self, x: torch.Tensor) -> torch.Tensor:
        """Psi of the node masses in GeV, beams in channel 1 and constituents in 0.

        The channel, not the value, is what tells a block that a node is a beam. A
        nominally massless constituent has m^2 at the float32 rounding of E^2 - |p|^2,
        which the absolute value keeps real.
        """
        mass = normsq4(x).abs().sqrt()
        if not self.add_beams:
            return psi(mass.unsqueeze(-1))
        scalars = x.new_zeros(*mass.shape, 2)
        scalars[:, :2, 1] = mass[:, :2]
        scalars[:, 2:, 0] = mass[:, 2:]
        return psi(scalars)

    def _message_passing(
        self, x: torch.Tensor, scalars: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the blocks over the flattened (jets * nodes) axis; return the scalars."""
        src, dst = self._batch_edges(node_mask)
        attr, flat_mask = scalars.reshape(-1, self.n_scalar), node_mask.reshape(-1)
        h, x = self.embedding(attr), x.reshape(-1, 4)
        for block in self.blocks:
            h, x = block(h, x, src, dst, attr, flat_mask)
        return h

    def _batch_edges(
        self, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Index the edges that join two real nodes, over the flattened node axis.

        Padded constituents are dropped to reduce the cost of the graphs.

        :param node_mask: (jets, nodes) boolean, true on a real node.
        :return: two (edges,) index tensors into the flattened jets * nodes axis.
        """
        valid = node_mask[:, self.edge_src] & node_mask[:, self.edge_dst]
        jet_idx, edge_idx = valid.nonzero(as_tuple=True)
        offset = jet_idx * node_mask.shape[1]
        return self.edge_src[edge_idx] + offset, self.edge_dst[edge_idx] + offset

    def _pool(self, h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Average the node scalars of each jet over its real nodes.

        Over the real nodes rather than the slots, unlike the reference implementation,
        so that widening the padding leaves the logits alone.
        """
        counts = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        masked = h * node_mask.reshape(-1, 1).to(h.dtype)
        return masked.view(node_mask.shape[0], -1, self.hidden_dim).sum(dim=1) / counts
