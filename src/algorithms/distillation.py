"""Distillation of a teacher into an MLP student, under the three objectives."""

import torch
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from torch import nn

from src.algorithms import BaseModule
from src.algorithms.classifier import Classifier
from src.algorithms.losses.hint import continued, hint_loss
from src.algorithms.losses.kd import kd_loss
from src.algorithms.losses.relational import relational_loss

OBJECTIVES = ("hint", "kd", "relational")


class DistillationModule(BaseModule):
    """Train a student against the teacher outputs cached for its fold.

    Batches carry ``(x, y, teacher_logits, teacher_hints)`` while fitting and
    validating, and ``(x, y)`` at test, where the teacher is run live instead.

    :param teacher_net: Hydra config of the teacher network, read for its hint width.
    :param teacher_ckpt: checkpoint of the trained teacher, loaded only at test.
    :param warm_start_ckpt: checkpoint of a hint run whose student and regressor open
        this one.
    :param guided_index: hidden layer of the student that reproduces the hint.
    :param alpha: weight of the cross entropy beside the distillation term.
    :param beta: share of the gradient the hint term carries; 0 drops it.
    :param freeze_above_guided: keep the layers past the guided one, and the output
        layer, at their initialisation.
    """

    def __init__(
        self,
        net: dict,
        num_classes: int,
        input_shape: tuple[int, ...],
        optimizer: dict,
        teacher_net: dict,
        scheduler: dict | None = None,
        compile: bool = False,
        teacher_ckpt: str | None = None,
        warm_start_ckpt: str | None = None,
        objective: str = "kd",
        guided_index: int = 1,
        temperature: float = 4.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        discrepancy: str = "cosine",
        lambda_is: float = 1.0,
        lambda_ic: float = 0.0,
        freeze_above_guided: bool = False,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {objective!r}"
            )
        super().__init__(net, num_classes, input_shape, optimizer, scheduler, compile)
        self.tracks_accuracy = objective != "hint"
        self.regressor = self._build_regressor(teacher_net)
        # A list, so the teacher is neither a submodule, nor checkpointed, nor counted.
        self._teacher: list[Classifier] = []
        if freeze_above_guided:
            self._freeze_above_guided()
        if warm_start_ckpt:
            self._warm_start(warm_start_ckpt)

    def _build_regressor(self, teacher_net: dict) -> nn.Module:
        """Map the guided layer onto the teacher's hint, or nothing when the widths
        already agree."""
        hint_dim, guided_dim = self._teacher_hint_dim(teacher_net), self._guided_dim()
        if guided_dim == hint_dim:
            return nn.Identity()
        return nn.Linear(guided_dim, hint_dim)

    @staticmethod
    def _teacher_hint_dim(teacher_net: dict) -> int:
        """Hint width of the teacher, built once when the width follows its
        arguments."""
        declared = get_class(teacher_net["_target_"]).hint_dim
        return declared or instantiate(teacher_net).hint_dim

    def _guided_dim(self) -> int:
        """Width of the guided layer, read from a dry pass at the network's own
        width."""
        example = torch.randn(2, *self.net_input_shape())
        with torch.no_grad():
            _, hidden = self.net.forward_with_hidden(self.net.prepare(example))
        return hidden[self.hparams.guided_index].shape[-1]

    def _freeze_above_guided(self) -> None:
        """Hold the layers the hint objective never trains at their initialisation."""
        for layer in self.net.hidden[self.hparams.guided_index + 1 :]:
            layer.requires_grad_(False)
        self.net.output.requires_grad_(False)

    def _warm_start(self, path: str) -> None:
        """Open this run with the student and regressor a hint run left behind."""
        source = DistillationModule.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        self.net.load_state_dict(source.net.state_dict())
        self.regressor.load_state_dict(source.regressor.state_dict())

    def teacher(self) -> Classifier:
        """The frozen teacher, loaded on first use; needed at test time alone."""
        if not self._teacher:
            self._teacher.append(self._load_teacher())
        return self._teacher[0]

    def _load_teacher(self) -> Classifier:
        teacher = Classifier.load_from_checkpoint(
            self.hparams.teacher_ckpt, map_location=self.device, weights_only=False
        )
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is not None:
            teacher.bind(datamodule)
        return teacher.eval().requires_grad_(False)

    def model_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Run the student, its regressor and the objective over one batch."""
        x, y = batch[0], batch[1]
        prepared = self.net.prepare(self._select(x), self.norm_scales)
        logits, hidden = self.net.forward_with_hidden(prepared)
        guided = self.regressor(hidden[self.hparams.guided_index])
        t_logits, t_hints = self._teacher_outputs(batch)
        return {
            "loss": self._loss(logits, guided, y, t_logits, t_hints),
            "logits": logits,
            "preds": logits.argmax(dim=1),
            "targets": y.argmax(dim=1),
        }

    def _teacher_outputs(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The cached teacher outputs of the batch, recomputed live when absent."""
        if len(batch) >= 4:
            return batch[2], batch[3]
        with torch.no_grad():
            return self.teacher().forward_with_hint(batch[0])

    def _loss(
        self,
        logits: torch.Tensor,
        guided: torch.Tensor,
        y: torch.Tensor,
        t_logits: torch.Tensor,
        t_hints: torch.Tensor,
    ) -> torch.Tensor:
        if self.hparams.objective == "hint":
            return hint_loss(guided, t_hints)
        main = self._main_loss(logits, y, t_logits)
        if self.hparams.beta <= 0.0:
            return main
        return continued(main, hint_loss(guided, t_hints), self.hparams.beta)

    def _main_loss(
        self, logits: torch.Tensor, y: torch.Tensor, t_logits: torch.Tensor
    ) -> torch.Tensor:
        """The distillation term of the chosen objective, plus alpha times the cross
        entropy."""
        if self.hparams.objective == "kd":
            main = kd_loss(logits, t_logits, self.hparams.temperature)
        else:
            main = relational_loss(
                logits,
                t_logits,
                self.hparams.discrepancy,
                self.hparams.lambda_is,
                self.hparams.lambda_ic,
            )
        if self.hparams.alpha > 0.0:
            main = main + self.hparams.alpha * F.cross_entropy(logits, y.argmax(dim=1))
        return main

    def _update_metrics(self, stage: str, out: dict[str, torch.Tensor]) -> None:
        """Add the hint objective's own monitor to the shared metrics."""
        super()._update_metrics(stage, out)
        if self.hparams.objective == "hint":
            self.log(
                f"{stage}/hint_mse",
                out["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
