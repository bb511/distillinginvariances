import torch
import torch.nn as nn

def activ_string_to_torch(activ: str):
    activations = {'relu': lambda: nn.ReLU(inplace=True), 'tanh': lambda: nn.Tanh(), 'sigmoid': lambda: nn.Sigmoid(), 'leaky_relu': lambda: nn.LeakyReLU()}
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(f'Activation {activ} not implemented.')
    return activation

class DeepSetsInvariant(nn.Module):

    def __init__(self, input_dim: int, phi_layers: list, rho_layers: list, activ: str, aggreg: str, dropout: float, output_dim: int):
        super().__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = list(phi_layers)
        self.rho_layers = list(rho_layers)
        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self) -> nn.Sequential:
        phi = nn.Sequential()
        layers = [self.input_dim] + self.phi_layers
        for nlayer in range(len(layers) - 1):
            phi.append(nn.Linear(layers[nlayer], layers[nlayer + 1]))
            phi.append(activ_string_to_torch(self.activ))
        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        rho = nn.Sequential()
        layers = [self.phi_layers[-1]] + self.rho_layers + [self.output_dim]
        for nlayer in range(len(layers) - 1):
            if 0 < dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(nn.Linear(layers[nlayer], layers[nlayer + 1]))
            if nlayer == len(layers) - 2:
                break
            rho.append(activ_string_to_torch(self.activ))
        return rho

    def _get_aggregation(self, aggreg: str):
        aggregations = {'mean': lambda: torch.mean, 'max': lambda: torch.max}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(f'Aggregation {aggreg} not implemented.')
        return aggregation

    def forward(self, x):
        phi_output = self.phi(x)
        agg_output = self.agg(phi_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        return self.rho(agg_output)

    def forward_with_hint(self, x):
        phi_output = self.phi(x)
        hint = self.agg(phi_output, dim=1)
        if isinstance(hint, tuple):
            hint = hint[0]
        logits = self.rho(hint)
        return (logits, hint)

    def forward_with_two_hints(self, x):
        phi_output = self.phi(x)
        hint1 = self.agg(phi_output, dim=1)
        if isinstance(hint1, tuple):
            hint1 = hint1[0]
        logits = self.rho(hint1)
        h = hint1
        hint2 = None
        for layer in self.rho:
            h = layer(h)
            if not isinstance(layer, (nn.Linear, nn.Dropout)):
                hint2 = h
                break
        return (logits, hint1, hint2)

    def forward_with_phi_hint(self, x, phi_depth: int=3):
        h = x
        hint = None
        target_idx = phi_depth * 2 - 1
        for i, layer in enumerate(self.phi):
            h = layer(h)
            if i == target_idx:
                hint = self.agg(h, dim=1)
                if isinstance(hint, tuple):
                    hint = hint[0]
        full_agg = self.agg(h, dim=1)
        if isinstance(full_agg, tuple):
            full_agg = full_agg[0]
        logits = self.rho(full_agg)
        return (logits, hint)

class MLPBasic(nn.Module):

    def __init__(self, input_dim: int, layers: list, output_dim: int, activ: str):
        super().__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.layers = list(layers)
        self.output_dim = output_dim
        self._construct_mlp()

    def _construct_mlp(self):
        all_layers = [self.input_dim] + self.layers + [self.output_dim]
        self.mlp = nn.Sequential()
        for nlayer in range(len(all_layers) - 1):
            self.mlp.add_module(f'linear_{nlayer}', nn.Linear(all_layers[nlayer], all_layers[nlayer + 1]))
            if nlayer < len(all_layers) - 2:
                self.mlp.add_module(f'activation_{nlayer}', activ_string_to_torch(self.activ))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

    def forward_with_guided(self, x, guided_idx: int=3):
        x = torch.flatten(x, start_dim=1)
        guided = None
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == guided_idx:
                guided = x
        return (x, guided)

    def forward_with_two_guided(self, x, guided_idx1: int, guided_idx2: int):
        x = torch.flatten(x, start_dim=1)
        guided1 = guided2 = None
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == guided_idx1:
                guided1 = x
            if i == guided_idx2:
                guided2 = x
        return (x, guided1, guided2)
