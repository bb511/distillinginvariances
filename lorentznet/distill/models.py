import math
import torch
from torch import nn
NORMALIZATIONS = ('batch', 'layer', 'none')

def feature_norm(kind: str, hidden_dim: int) -> nn.Module:
    if kind == 'batch':
        return nn.BatchNorm1d(hidden_dim)
    if kind == 'layer':
        return nn.LayerNorm(hidden_dim)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f'normalization must be one of {NORMALIZATIONS}, got {kind!r}')

def psi(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def normsq4(x: torch.Tensor) -> torch.Tensor:
    squared = x.square()
    return 2.0 * squared[..., 0] - squared.sum(dim=-1)

def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    product = p * q
    return 2.0 * product[..., 0] - product.sum(dim=-1)

class LGEB(nn.Module):

    def __init__(self, hidden_dim: int, n_scalar: int, c_weight: float, normalization: str='batch', last_layer: bool=False) -> None:
        super().__init__()
        dim = hidden_dim
        self.c_weight = c_weight
        self.last_layer = last_layer
        self.phi_e = nn.Sequential(nn.Linear(2 * dim + 2, dim, bias=False), feature_norm(normalization, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.phi_m = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        if not last_layer:
            coordinate_out = nn.Linear(dim, 1, bias=False)
            nn.init.xavier_uniform_(coordinate_out.weight, gain=0.001)
            self.phi_x = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), coordinate_out)
        self.phi_h = nn.Sequential(nn.Linear(dim + dim + n_scalar, dim), feature_norm(normalization, dim), nn.ReLU(), nn.Linear(dim, dim))

    @staticmethod
    def _apply_flat(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        return module(value.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, node_attr: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, dim = h.shape
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        no_self = ~torch.eye(nodes, dtype=torch.bool, device=x.device).unsqueeze(0)
        edge_mask = pair_mask & no_self
        edge_mask_f = edge_mask.unsqueeze(-1).to(x.dtype)
        xi = x.unsqueeze(2)
        xj = x.unsqueeze(1)
        difference = xi - xj
        edge_input = torch.cat([h.unsqueeze(2).expand(batch, nodes, nodes, dim), h.unsqueeze(1).expand(batch, nodes, nodes, dim), psi(normsq4(difference)).unsqueeze(-1), psi(dotsq4(xi, xj)).unsqueeze(-1)], dim=-1)
        valid_message = self.phi_e(edge_input[edge_mask])
        valid_message = valid_message * self.phi_m(valid_message)
        message = valid_message.new_zeros(batch, nodes, nodes, dim)
        message[edge_mask] = valid_message
        if not self.last_layer:
            valid_attention = self.phi_x(valid_message)
            attention = valid_attention.new_zeros(batch, nodes, nodes, 1)
            attention[edge_mask] = valid_attention
            translation = (difference * attention).clamp(-100.0, 100.0)
            translation = translation * edge_mask_f
            neighbours = edge_mask.sum(dim=2, keepdim=True).clamp(min=1)
            x = x + self.c_weight * translation.sum(dim=2) / neighbours
        aggregate = (message * edge_mask_f).sum(dim=2)
        node_input = torch.cat([h, aggregate, node_attr], dim=-1)
        valid_update = self.phi_h(node_input[node_mask])
        update = valid_update.new_zeros(batch, nodes, dim)
        update[node_mask] = valid_update
        h = h + update
        return (x, h)

class LorentzNet(nn.Module):

    def __init__(self, hidden_dim: int=96, n_layers: int=4, n_classes: int=5, c_weight: float=0.001, dropout: float=0.2, add_beams: bool=True, beam_mass: float=1.0, normalization: str='batch') -> None:
        super().__init__()
        if normalization not in NORMALIZATIONS:
            raise ValueError(f'normalization must be one of {NORMALIZATIONS}, got {normalization!r}')
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.add_beams = add_beams
        self.beam_mass = beam_mass
        self.normalization = normalization
        self.n_scalar = 2 if add_beams else 1
        self.embedding = nn.Linear(self.n_scalar, hidden_dim)
        self.blocks = nn.ModuleList([LGEB(hidden_dim, self.n_scalar, c_weight, normalization=normalization, last_layer=index == n_layers - 1) for index in range(n_layers)])
        self.decode = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def _prepare(self, x: torch.Tensor, lorentz_matrix: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if self.add_beams:
            beam_energy = math.sqrt(1.0 + self.beam_mass ** 2)
            beams = x.new_tensor([[[beam_energy, 0.0, 0.0, 1.0], [beam_energy, 0.0, 0.0, -1.0]]]).expand(batch, 2, 4)
            if lorentz_matrix is not None:
                beams = beams @ lorentz_matrix.transpose(-1, -2)
            x = torch.cat([beams, x], dim=1)
            mass = normsq4(x).abs().sqrt()
            scalars = x.new_zeros(batch, x.shape[1], 2)
            scalars[:, :2, 1] = mass[:, :2]
            scalars[:, 2:, 0] = mass[:, 2:]
            scalars = psi(scalars)
        else:
            scalars = psi(normsq4(x).abs().sqrt().unsqueeze(-1))
        node_mask = x[..., 0] != 0.0
        return (x, scalars, node_mask)

    def forward_with_hint(self, x: torch.Tensor, lorentz_matrix: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        x, scalars, node_mask = self._prepare(x, lorentz_matrix)
        mask = node_mask.unsqueeze(-1).to(x.dtype)
        h = self.embedding(scalars) * mask
        for block in self.blocks:
            x, h = block(x, h, scalars, node_mask)
            h = h * mask
        pooled = (h * mask).mean(dim=1)
        return (self.decode(pooled), pooled)

    def forward(self, x: torch.Tensor, lorentz_matrix: torch.Tensor | None=None) -> torch.Tensor:
        return self.forward_with_hint(x, lorentz_matrix)[0]

class MLPStudent(nn.Module):

    def __init__(self, hidden: list[int] | tuple[int, ...]=(64, 32, 32, 32)) -> None:
        super().__init__()
        sizes = [96, *hidden]
        self.hidden = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(hidden))])
        self.output = nn.Linear(hidden[-1], 5)

    def forward_with_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        value = x.flatten(start_dim=1)
        activations = []
        for layer in self.hidden:
            value = torch.relu(layer(value))
            activations.append(value)
        return (self.output(value), activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(x)[0]

class HintModel(nn.Module):

    def __init__(self, student: MLPStudent, guided_hidden_index: int, hint_dim: int=96) -> None:
        super().__init__()
        self.student = student
        self.guided_hidden_index = guided_hidden_index
        guided_dim = student.hidden[guided_hidden_index].out_features
        self.projector = nn.Identity() if guided_dim == hint_dim else nn.Linear(guided_dim, hint_dim)

    def guided(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.student.forward_with_hidden(x)
        return self.projector(hidden[self.guided_hidden_index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)
