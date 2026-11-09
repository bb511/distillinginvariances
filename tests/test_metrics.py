import torch

from src.metrics.calibration import ECEAccumulator
from src.metrics.divergence import jsd
from src.metrics.fidelity import pair_means, sqrt_jsd
from src.metrics.invariance import invariance
from src.metrics.temperature import fit_temperature, scale


def test_jsd_and_the_pair_measures_at_their_ends() -> None:
    one_hot = torch.eye(5)
    p = torch.randn(64, 5, generator=torch.Generator().manual_seed(0)).softmax(-1)

    assert float(jsd(p, p).abs().max()) < 1e-12
    assert torch.allclose(jsd(one_hot, one_hot.roll(1, dims=0)), torch.ones(5))
    assert pair_means(one_hot * 200, one_hot * 200)[:2] == (1.0, 0.0)
    top1, distance, legacy = pair_means(one_hot * 200, one_hot.roll(1, dims=0) * 200)
    assert top1 == 0.0 and abs(distance - 1.0) < 1e-9 and abs(legacy) < 1e-9


def test_sqrt_jsd_is_a_metric_and_each_side_takes_its_own_temperature() -> None:
    generator = torch.Generator().manual_seed(1)
    a, b, c = (torch.randn(4096, 5, generator=generator) * 3.0 for _ in range(3))

    assert bool((sqrt_jsd(a, c) <= sqrt_jsd(a, b) + sqrt_jsd(b, c) + 1e-12).all())
    assert float(sqrt_jsd(a, 2.0 * a, 1.0, 2.0).max()) < 1e-7
    assert float(sqrt_jsd(a, 2.0 * a).min()) > 0.0


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


def test_invariance_averages_over_every_draw() -> None:
    clean = torch.randn(16, 5, generator=torch.Generator().manual_seed(2))
    flipped = clean.roll(1, dims=1)
    draws = torch.stack([clean, clean, flipped, flipped])

    top1, distance, legacy = invariance(clean, draws[:2])
    assert (top1, distance, round(legacy, 9)) == (1.0, 0.0, 1.0)
    assert invariance(clean, draws)[0] == 0.5 * (1.0 + pair_means(clean, flipped)[0])


def test_fit_temperature_recovers_a_planted_temperature() -> None:
    # Labels drawn from softmax(logits) make the optimum T = 1 for the unscaled logits.
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(200_000, 5, generator=generator) * 2.0
    labels = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)

    assert abs(fit_temperature(scale(logits, 1 / 2.5), labels) / 2.5 - 1.0) < 0.01
