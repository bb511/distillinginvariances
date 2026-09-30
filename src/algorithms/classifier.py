"""Cross-entropy training: the teachers and the from-scratch baselines."""

import torch
import torch.nn.functional as F

from src.algorithms import BaseModule


class Classifier(BaseModule):
    """Train a network on the labels alone."""

    def model_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Return the loss, the logits, the predictions and the integer targets.

        :param batch:``(x, y)``, with *y* one-hot float32 (batch, classes) from both
            datamodules.
        """
        x, y = batch[0], batch[1]
        targets = y.argmax(dim=1)
        logits = self(x)
        return {
            "loss": F.cross_entropy(logits, targets),
            "logits": logits,
            "preds": logits.argmax(dim=1),
            "targets": targets,
        }
