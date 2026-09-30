import torch

from src.metrics.calibration import ECEAccumulator
from src.metrics.divergence import jsd
from src.metrics.fidelity import fidelity
from src.metrics.invariance import invariance
from src.metrics.temperature import fit_temperature, scale


def test_jsd_and_fidelity_at_their_ends() -> None:
    one_hot = torch.eye(5)
    p = torch.randn(64, 5, generator=torch.Generator().manual_seed(0)).softmax(-1)

    assert float(jsd(p, p).abs().max()) < 1e-12
    assert torch.allclose(jsd(one_hot, one_hot.roll(1, dims=0)), torch.ones(5))
    assert fidelity(one_hot * 200, one_hot * 200)[0] == 1.0
    top1, similarity = fidelity(one_hot * 200, one_hot.roll(1, dims=0) * 200)
    assert top1 == 0.0 and abs(similarity) < 1e-9


def test_ece_matches_a_naive_reference() -> None:
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(4096, 5, generator=generator) * 2.0
    labels = torch.randint(0, 5, (4096,), generator=generator)
    confidence, prediction = logits.double().softmax(-1).max(-1)
    accumulator = ECEAccumulator()
    for start in range(0, 4096, 500):
        accumulator.update(logits[start : start + 500], labels[start : start + 500])

    naive = 0.0
    for b in range(15):
        inside = (confidence * 15).clamp(max=14).long() == b
        if inside.any():
            gap = (
                prediction[inside].eq(labels[inside]).double().mean()
                - confidence[inside].mean()
            )
            naive += float(inside.sum()) / 4096 * abs(float(gap))

    assert abs(accumulator.compute() - naive) < 1e-9


def test_invariance_of_an_identity_draw_is_one() -> None:
    net = torch.nn.Linear(4, 5)
    top1, similarity = invariance(
        lambda x, _: net(x), torch.randn(16, 4), lambda x: (x, None), 3
    )

    assert (top1, round(similarity, 9)) == (1.0, 1.0)


def test_fit_temperature_recovers_a_planted_temperature() -> None:
    # Labels drawn from softmax(logits) make the optimum T = 1 for the unscaled logits.
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(200_000, 5, generator=generator) * 2.0
    labels = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)

    assert abs(fit_temperature(scale(logits, 1 / 2.5), labels) / 2.5 - 1.0) < 0.01
