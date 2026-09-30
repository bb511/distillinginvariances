"""The Hinton knowledge-distillation term."""

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Return :math:`T^2 \\, \\mathrm{KL}(p_T \\Vert p_S)` at temperature *T*.

    The T^2 factor keeps the gradient of the soft term comparable with the cross entropy
    as T varies (arXiv:1503.02531, section 2.1). Both arguments are logits, shape
    (batch, classes), and the reduction is the mean over the batch.
    """
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
