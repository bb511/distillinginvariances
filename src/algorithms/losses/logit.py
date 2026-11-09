"""Logit matching: the student regresses the teacher's raw logits."""

import torch
import torch.nn.functional as F


def logit_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Mean squared error between the two sets of logits, with no temperature.

    Equation (1) of Ba and Caruana (arXiv:1312.6184, section 2.2) times 2 / classes:
    both arguments are (batch, classes), and the mean runs over both axes.
    """
    return F.mse_loss(student_logits, teacher_logits)
