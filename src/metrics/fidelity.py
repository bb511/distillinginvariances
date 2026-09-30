"""How closely a student reproduces its teacher on the same rows."""

import torch

from src.metrics.divergence import jsd


def fidelity(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> tuple[float, float]:
    """Top-1 agreement and ``1 - mean JSD`` over two paired ``(rows, classes)`` tensors.

    The softmaxes are taken in float64: the mean runs over a whole split, and the study
    quotes the third digit of a number close to 1.
    """
    top1 = student_logits.argmax(-1).eq(teacher_logits.argmax(-1)).double().mean()
    divergence = jsd(
        student_logits.double().softmax(-1), teacher_logits.double().softmax(-1)
    )
    return float(top1), float(1.0 - divergence.mean())
