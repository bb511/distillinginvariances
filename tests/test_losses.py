import torch
import torch.nn.functional as F

from src.algorithms.losses.hint import continued, hint_loss
from src.algorithms.losses.kd import kd_loss
from src.algorithms.losses.relational import discrepancy, edges, relational_loss


def _logits(rows: int, classes: int, seed: int) -> torch.Tensor:
    return torch.randn(rows, classes, generator=torch.Generator().manual_seed(seed))


def test_kd_is_temperature_squared_times_kl() -> None:
    student, teacher, temperature = _logits(6, 5, 0), _logits(6, 5, 1), 4.0
    p_teacher, p_student = F.softmax(teacher / temperature, -1), F.softmax(
        student / temperature, -1
    )
    kl = (p_teacher * (p_teacher.log() - p_student.log())).sum(-1).mean()

    assert torch.allclose(kd_loss(student, teacher, temperature), temperature**2 * kl)


def test_the_hint_terms() -> None:
    guided, hint = _logits(4, 32, 2), _logits(4, 32, 3)
    main = torch.tensor(2.0)

    assert torch.allclose(hint_loss(guided, hint), ((guided - hint) ** 2).mean())
    assert continued(main, torch.tensor(0.5), 0.0) == main
    # The hint is rescaled to the size of the main term, so the mix stays at that size.
    assert torch.allclose(continued(main, torch.tensor(0.5), 0.25), main)


def test_the_relational_terms() -> None:
    student, teacher = _logits(6, 5, 8), _logits(6, 5, 9)
    ds, dt = edges(student), edges(teacher)
    cosine = F.cosine_similarity(ds, dt, dim=-1, eps=1e-8)

    assert (
        ds.shape == (30, 5) and (ds.abs().sum(-1) > 0).all()
    )  # both orderings, no self-pair
    assert torch.allclose(discrepancy(ds, dt, "cosine"), (1.0 - cosine).mean())
    assert all(
        discrepancy(dt, dt, kind) < 1e-6
        for kind in ("mse", "smooth_l1", "distance_mse")
    )
    inter_class = discrepancy(edges(student.T), edges(teacher.T), "cosine")
    expected = 2.0 * discrepancy(ds, dt, "cosine") + 3.0 * inter_class
    assert torch.allclose(
        relational_loss(student, teacher, "cosine", 2.0, 3.0), expected
    )
    assert relational_loss(student[:1], teacher[:1], "cosine") == 0.0
