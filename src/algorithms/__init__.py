"""Training logic shared by the classifiers and the distilled students."""

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from src.algorithms.utils.flops import record_flops
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def resolve_feature_indices(
    model_features: list[str], feature_names: list[str]
) -> list[int]:
    """Feature indices of *model_features* inside *feature_names*.

    Names match bare or with the record's ``j1_`` prefix, so ``px`` finds ``j1_px``.
    """
    index_of = {name: idx for idx, name in enumerate(feature_names)}
    missing = [
        f for f in model_features if f not in index_of and f"j1_{f}" not in index_of
    ]
    if missing:
        raise ValueError(
            f"Feature {missing[0]!r} is absent from the data columns {feature_names}. "
            f"The network reads {', '.join(model_features)}."
        )
    return [index_of.get(f, index_of.get(f"j1_{f}")) for f in model_features]


class BaseModule(LightningModule):
    """A network, its metrics and its optimiser, over batches of the full stored width.

    The network, the optimiser and the scheduler arrive as plain dicts and are
    instantiated here, so `load_from_checkpoint` rebuilds a module with no Hydra config
    at hand.

    :param net: Hydra config of a `src.algorithms.components.FeatureNet`.
    :param input_shape: shape of one stored sample, without the batch axis.
    :param optimizer: Hydra config of a partial optimiser, called with ``params``.
    :param scheduler: Lightning's scheduler dict, its ``scheduler`` entry a partial.
    """

    # Resolved in bind() from the datamodule; None for MNIST, which has no named columns.
    model_feature_indices: list[int] | None = None
    norm_scales: dict[str, float] | None = None
    # The hint objective has no class prediction to score.
    tracks_accuracy: bool = True

    def __init__(
        self,
        net: dict,
        num_classes: int,
        input_shape: tuple[int, ...],
        optimizer: dict,
        scheduler: dict | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = instantiate(net)
        self._build_metrics(num_classes)
        self.example_input_array = torch.randn(2, *input_shape)
        record_flops(self, self.net_input_shape())

    def _build_metrics(self, num_classes: int) -> None:
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def net_input_shape(self) -> tuple[int, ...]:
        """Shape of one sample once the network's own columns have been selected."""
        shape = tuple(self.hparams.input_shape)
        required = self.net.required_features
        return shape if required is None else (*shape[:-1], len(required))

    def bind(self, datamodule) -> None:
        """Read from *datamodule* which columns the network gets and at which scale.

        Called from `setup` under a trainer, and directly by the evaluation entry
        points, which run the module without one.
        """
        feature_names = getattr(datamodule, "feature_names", None)
        required = self.net.required_features
        if feature_names is not None and required:
            self.model_feature_indices = resolve_feature_indices(
                required, list(feature_names)
            )
        self.norm_scales = datamodule.norm_scales

    def setup(self, stage: str) -> None:
        """Bind to the datamodule, and compile the network."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        # _trainer rather than trainer: the public property raises when the module
        # stands alone.
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is not None:
            self.bind(datamodule)

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        """Cast a batch to float32 and reduce it to the columns the network reads.

        The guard is the width, so a batch already reduced passes through untouched.
        """
        x = x.to(torch.float32)
        indices = self.model_feature_indices
        if indices is not None and x.shape[-1] != len(indices):
            return x[..., indices]
        return x

    def forward(self, x: torch.Tensor, **net_kwargs) -> torch.Tensor:
        """Return the class logits of a batch of the full stored width."""
        prepared = self.net.prepare(self._select(x), self.norm_scales)
        return self.net(prepared, **net_kwargs)

    def forward_with_hint(
        self, x: torch.Tensor, **net_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the network's hint, selecting columns as `forward`."""
        prepared = self.net.prepare(self._select(x), self.norm_scales)
        return self.net.forward_with_hint(prepared, **net_kwargs)

    def model_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Run one forward pass and return at least ``loss``; every subclass has one."""
        raise NotImplementedError(f"{type(self).__name__} must implement model_step().")

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int):
        out = self.model_step(batch)
        self._update_metrics("train", out)
        return out["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        out = self.model_step(batch)
        self._update_metrics("val", out)
        return out

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        out = self.model_step(batch)
        self._update_metrics("test", out)
        return out

    def _update_metrics(self, stage: str, out: dict[str, torch.Tensor]) -> None:
        """Log the epoch means under the names the configs monitor."""
        loss_metric = getattr(self, f"{stage}_loss")
        loss_metric(out["loss"])
        self.log(
            f"{stage}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True
        )
        if not self.tracks_accuracy:
            return
        acc_metric = getattr(self, f"{stage}_acc")
        acc_metric(out["preds"], out["targets"])
        self.log(
            f"{stage}/acc", acc_metric, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_train_start(self) -> None:
        """Clean up the sanity check values."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def on_train_epoch_end(self) -> None:
        line = (
            f"[epoch {self.current_epoch:>3d}] train | "
            f"loss {self.train_loss.compute():.4f}"
        )
        if self.tracks_accuracy:
            line += f" acc {self.train_acc.compute():.4f}"
        log.info(line)

    def on_validation_epoch_end(self) -> None:
        if not self.tracks_accuracy:
            log.info(
                f"[epoch {self.current_epoch:>3d}]   val | "
                f"loss {self.val_loss.compute():.4f}"
            )
            return
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        best = self.val_acc_best.compute()
        self.log("val/acc_best", best, sync_dist=True, prog_bar=True)
        log.info(
            f"[epoch {self.current_epoch:>3d}]   val | "
            f"loss {self.val_loss.compute():.4f} acc {acc:.4f} best {best:.4f}"
        )

    def configure_optimizers(self) -> dict:
        """Build the optimiser over the trainable parameters, and its scheduler."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        # _partial_=False: the parameters are known only here.
        optimizer = instantiate(
            self.hparams.optimizer, params=trainable, _partial_=False
        )
        if not self.hparams.scheduler:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": self._set_up_scheduler(optimizer),
        }

    def _set_up_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Instantiate the scheduler partial inside the dict Lightning expects.

        A schedule that needs the run's length reads ``trainer.max_epochs`` through its
        own config, so nothing is injected here.
        """
        scheduler_cfg = dict(self.hparams.scheduler)
        scheduler_cfg["scheduler"] = instantiate(
            scheduler_cfg["scheduler"], optimizer=optimizer, _partial_=False
        )
        return scheduler_cfg
