import numpy as np
import torch

from src.algorithms.components.mlp import MLPStudent
from src.data.components.augmentation import apply_permutation
from src.metrics import profile as orbits
from tests.conftest import compose_root


class StubModule(torch.nn.Module):
    """What layer_outputs reads of a BaseModule: net, _select, norm_scales and the hint
    forward."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net, self.norm_scales = net, None

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.forward_with_hint(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SumPooledNet(torch.nn.Module):
    hint_dim = 6

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.phi, self.rho = torch.nn.Linear(4, 6), torch.nn.Linear(6, 3)

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.relu(self.phi(x)).sum(1)
        return self.rho(pooled), pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(x)[0]


def _fraction(acts: torch.Tensor, exhaustive: bool = False) -> float:
    moments = orbits.sample_moments(acts)
    return orbits.invariance_fraction(moments, acts.shape[1], exhaustive, 50, 0)[
        "invariance"
    ]


def test_the_fraction_is_one_on_orbits_and_near_zero_across_draws_alone() -> None:
    generator = torch.Generator().manual_seed(0)
    per_sample = torch.randn(400, 1, 5, generator=generator).expand(400, 8, 5)
    per_draw = torch.randn(400, 8, 5, generator=generator)

    assert abs(_fraction(per_sample) - 1.0) < 1e-12
    assert abs(_fraction(per_draw)) < 0.05


def test_an_exactly_invariant_teacher_profiles_at_one() -> None:
    module = StubModule(SumPooledNet())
    x = torch.randn(64, 10, 4, generator=torch.Generator().manual_seed(1))
    orbit = torch.stack([apply_permutation(x)[0] for _ in range(6)], 1)

    outputs = orbits.orbit_outputs(module, orbit)

    assert list(outputs) == ["hint", "logits"]
    assert all(abs(_fraction(acts) - 1.0) < 1e-6 for acts in outputs.values())


def test_a_student_names_its_hidden_layers_and_is_not_invariant() -> None:
    module = StubModule(MLPStudent(in_dim=40, hidden=(8, 8), output_dim=3))
    x = torch.randn(64, 10, 4, generator=torch.Generator().manual_seed(2))
    orbit = torch.stack([apply_permutation(x)[0] for _ in range(6)], 1)

    outputs = orbits.orbit_outputs(module, orbit)

    assert list(outputs) == ["hidden_1", "hidden_2", "logits"]
    assert outputs["hidden_1"].shape == (64, 6, 8)
    assert _fraction(outputs["hidden_1"]) < 0.9


def test_the_torus_spectrum_puts_an_invariant_unit_at_zero_frequency() -> None:
    images = torch.rand(5, 1, 6, 6, generator=torch.Generator().manual_seed(3))
    orbit = orbits.torus_orbit(images)
    ink = orbit.flatten(2).sum(-1, keepdim=True)
    corner = orbit[:, :, 0, 0, :1]

    flat = orbits.torus_spectrum(ink, 6)
    shares = orbits.spectrum_shares(
        orbits.torus_spectrum(corner, 6), orbits.sample_moments(corner)
    )
    sums = orbits.shared_draw_sums(ink)

    assert torch.equal(orbit[:, 0], images) and orbit.shape == (5, 36, 1, 6, 6)
    assert abs(flat.sum() - flat[0, 0]) < 1e-9 * flat.sum()
    assert abs(shares.sum() - 1.0) < 1e-9
    assert orbits.normalised_variance(orbits.sample_moments(ink), sums) < 1e-9


def test_the_transposition_map_locates_the_slot_a_unit_reads() -> None:
    x = torch.randn(32, 5, 2, generator=torch.Generator().manual_seed(4))
    orbit = orbits.transposition_orbits(x)
    clean = x[:, 0, :1]

    moved, scale = orbits.transposition_distances(orbit[:, :, 0, :1], clean)
    matrix = orbits.pair_matrix(moved / scale, 5)

    assert torch.equal(orbit[:, 0, 0], x[:, 1]) and torch.equal(orbit[:, 0, 1], x[:, 0])
    assert np.allclose(matrix, matrix.T) and np.all(matrix[0, 1:] > 0)
    assert np.all(matrix[1:, 1:] == 0)


def test_the_profile_config_composes() -> None:
    cfg = compose_root("orbit_profile.yaml", "experiment=mnist_cnn", "method=kd")

    assert (cfg.task_name, cfg.eval_split, cfg.random_init) == (
        "profile",
        "test",
        False,
    )
    assert cfg.checkpoint_manifest_path.endswith("checkpoint_manifest.json")
    assert cfg.results.path.endswith("profile.csv")
