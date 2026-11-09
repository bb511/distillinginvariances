"""Cross-entropy training: teachers, baselines and label-smoothed students."""

import torch
import torch.nn.functional as F

from src.algorithms import BaseModule


class Classifier(BaseModule):
    """Train a network on the labels alone.

    :param label_smoothing: weight of the uniform distribution mixed into the one-hot
        targets. Its default is what loads a checkpoint whose hparams predate it.
    """

    def __init__(
        self,
        net: dict,
        num_classes: int,
        input_shape: tuple[int, ...],
        optimizer: dict,
        scheduler: dict | None = None,
        compile: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(net, num_classes, input_shape, optimizer, scheduler, compile)

    def model_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Return the loss, the logits, the predictions and the integer targets.

        :param batch:``(x, y)``, with *y* one-hot float32 (batch, classes) from both
            datamodules.
        """
        x, y = batch[0], batch[1]
        targets = y.argmax(dim=1)
        logits = self(x)
        smoothing = self.hparams.label_smoothing
        return {
            "loss": F.cross_entropy(logits, targets, label_smoothing=smoothing),
            "logits": logits,
            "preds": logits.argmax(dim=1),
            "targets": targets,
        }
