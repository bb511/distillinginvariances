import math

import torch

from src.algorithms.components.mlp import MLPStudent
from src.data.components.augmentation import (
    _sample_boosts,
    apply_permutation,
    sample_z_matrices,
    translate,
)
from src.metrics.concepts import concept_labels, frame_labels
from src.metrics.erasure import (
    erase,
    fit_probe,
    forward_from,
    leace,
    r_squared,
    random_removal,
    ridge_predict,
)
from src.metrics.magnitude import magnitudes
from tests.conftest import compose_root

NAMES = ["pt", "etarel", "phirel", "e", "px", "py", "pz"]


def _linear_concept(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4000, 12, generator=generator, dtype=torch.float64)
    mixing = torch.randn(12, 2, generator=generator, dtype=torch.float64)
    z = x @ mixing + 0.1 * torch.randn(
        4000, 2, generator=generator, dtype=torch.float64
    )
    return x, z


def test_a_ridge_probe_recovers_a_linear_concept() -> None:
    x, z = _linear_concept(0)

    predicted = ridge_predict(fit_probe(x[:3000], z[:3000]), x[3000:])

    assert float(r_squared(z[3000:], predicted)) > 0.95


def test_leace_leaves_no_linear_trace_and_moves_little() -> None:
    x, z = _linear_concept(1)

    erased = erase(x, leace(x, z))
    cross = (erased - erased.mean(0)).T @ (z - z.mean(0)) / len(x)
    refit = ridge_predict(fit_probe(erased[:3000], z[:3000]), erased[3000:])

    assert float(cross.abs().max()) < 1e-10
    assert float(r_squared(z[3000:], refit)) < 0.01
    assert torch.linalg.matrix_rank(erased - x).item() == 2


def test_the_random_control_removes_a_subspace_of_the_same_rank() -> None:
    x, _ = _linear_concept(2)

    mean, matrix = random_removal(x, 3, torch.Generator().manual_seed(0))

    assert (
        torch.allclose(matrix @ matrix, matrix)
        and abs(float(matrix.trace()) - 3) < 1e-9
    )


def test_the_frames_encode_each_draw() -> None:
    images = torch.zeros(2, 1, 28, 28)
    shifts = torch.tensor([[0, 7], [14, 21]])
    pt = torch.tensor([[5.0, 3.0, 1.0, 0.0], [0.0, 1.0, 3.0, 5.0]])
    jets = torch.zeros(2, 4, len(NAMES))
    jets[..., 0] = pt

    torus = frame_labels("translation", shifts, images, (None, None))
    slots = frame_labels("permutation", None, jets, (NAMES, None))
    beam = frame_labels(
        "z",
        torch.tensor([[math.pi, 0.5]], dtype=torch.float64),
        jets[:1],
        (NAMES, None),
    )

    assert torch.allclose(
        torus[1], torch.tensor([-1.0, 0.0, 0.0, -1.0]).double(), atol=1e-12
    )
    expected = torch.tensor([[-1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(slots, expected, atol=1e-12)
    assert torch.allclose(beam, torch.tensor([[-1.0, 0.0, 0.5]]).double(), atol=1e-12)


def test_the_so3_and_lorentz_iso_frames_encode_each_draw() -> None:
    half_turn = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    beam_to_x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rotations = torch.stack([torch.eye(3), half_turn, beam_to_x])
    torch.manual_seed(5)
    beam, draws = sample_z_matrices(8, 1.0)
    boosts = torch.cat([_sample_boosts(256, 1.0, True, None), torch.eye(4)[None]])
    layout = (None, None)

    line = frame_labels("so3", rotations, None, layout)
    along = frame_labels("lorentz_iso", beam, None, layout)
    vector = frame_labels("lorentz_iso", boosts, None, layout)
    sizes = magnitudes("lorentz_iso", boosts, None)

    expected = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, math.sqrt(3) / 2, -0.5]]
    assert torch.allclose(line, torch.tensor(expected).double(), atol=1e-7)
    assert torch.allclose(along[:, :2], torch.zeros(8, 2).double(), atol=1e-7)
    assert torch.allclose(along[:, 2], draws[:, 1].double(), atol=1e-6)
    norm = vector.norm(dim=1)
    assert torch.allclose(norm, sizes["rapidity"], atol=1e-6)
    assert torch.allclose(vector[:-1, 2] / norm[:-1], sizes["axis_cos"][:-1], atol=1e-5)
    assert torch.equal(vector[-1], torch.zeros(3).double())


def test_the_concepts_hold_on_their_orbits() -> None:
    images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(3))
    jets = torch.zeros(1, 3, len(NAMES))
    # Two massless constituents back to back in x, 10 GeV each: mass 20 GeV.
    jets[0, :2] = torch.tensor(
        [
            [10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            [10.0, 0.0, 3.0, 10.0, -10.0, 0.0, 0.0],
        ]
    )
    shifted = translate(images, torch.tensor([[3, 5], [27, 1], [14, 14]]))

    image, moved = concept_labels("mnist", images, None), concept_labels(
        "mnist", shifted, None
    )
    jet = concept_labels("jets", jets, (NAMES, None))
    permuted = concept_labels("jets", apply_permutation(jets)[0], (NAMES, None))
    top = concept_labels("top_tagging", jets, (NAMES, None))

    assert all(torch.allclose(image[k], moved[k]) for k in image)
    assert abs(float(jet["mass"]) - 20.0) < 1e-6 and float(jet["multiplicity"]) == 2.0
    assert all(torch.allclose(jet[k], permuted[k]) for k in jet)
    assert all(torch.equal(jet[k], top[k]) for k in jet)


def test_a_student_continues_from_any_hidden_layer() -> None:
    torch.manual_seed(4)
    net = MLPStudent(in_dim=10, hidden=(8, 6, 4), output_dim=3)
    x = torch.randn(16, 10)

    logits, hidden = net.forward_with_hidden(x)

    assert all(
        torch.allclose(forward_from(net, i, h), logits) for i, h in enumerate(hidden)
    )


def test_the_probe_config_composes() -> None:
    cfg = compose_root("concept_probe.yaml", "experiment=mnist_cnn", "method=kd")

    assert (cfg.task_name, cfg.eval_data, cfg.probe.n_draws) == (
        "probe",
        "canonical",
        1,
    )
    assert cfg.results.path.endswith("probes.csv")
