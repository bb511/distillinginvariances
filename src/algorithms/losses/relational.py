"""Relational distillation: the student reproduces the teacher's pairwise
differences."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def edges(z: torch.Tensor) -> torch.Tensor:
    """Pairwise differences :math:`z_i - z_j` of the rows of *z*.

    :param z: (nodes, dim), the batch of logits or its transpose.
    :return: (nodes * (nodes - 1), dim); the diagonal is dropped and both orderings of
        a pair are kept.
    """
    difference = z[:, None, :] - z[None, :, :]
    off_diagonal = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return difference[off_diagonal]


def discrepancy(ds: torch.Tensor, dt: torch.Tensor, kind: str) -> torch.Tensor:
    """Compare the student's edges *ds* with the teacher's *dt*, shapes (pairs, dim).

    Every scale is a detached teacher statistic, so the loss is dimensionless and its
    weight means the same thing across datasets and temperatures.

    :param kind: one of `src.vocabulary.DISCREPANCIES`.
    """
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(ds, dt, dim=-1, eps=1e-8)).mean()
    if kind == "mse":
        return _scaled(F.mse_loss, ds, dt, dt.square().mean().sqrt())
    if kind == "smooth_l1":
        return _scaled(F.smooth_l1_loss, ds, dt, dt.abs().mean())
    if kind == "distance_mse":
        student, teacher = ds.norm(dim=-1), dt.norm(dim=-1)
        return _scaled(F.mse_loss, student, teacher, teacher.mean())
    raise ValueError(f"Unknown relational discrepancy: {kind!r}")


def _scaled(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ds: torch.Tensor,
    dt: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply *loss_fn* to both arguments divided by the detached teacher *scale*."""
    scale = scale.detach().clamp_min(1e-8)
    return loss_fn(ds / scale, dt / scale)


def relational_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kind: str,
    lambda_is: float = 1.0,
    lambda_ic: float = 0.0,
) -> torch.Tensor:
    """Weighted sum of the inter-sample and inter-class relational terms.

    The inter-sample term compares the edges between the rows of the batch, the inter-
    class term those between the columns, which are the class responses over the batch.
    Every pair is used: nothing is subsampled.
    """
    if student_logits.shape[0] < 2:
        # One sample spans no edge, and the mean over an empty set would be a NaN.
        return student_logits.sum() * 0.0
    inter_sample = discrepancy(edges(student_logits), edges(teacher_logits), kind)
    inter_class = discrepancy(
        edges(student_logits.transpose(0, 1)),
        edges(teacher_logits.transpose(0, 1)),
        kind,
    )
    return lambda_is * inter_sample + lambda_ic * inter_class
