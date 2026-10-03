# Fit one stage of the distillation DAG and leave a manifest naming its checkpoint.

import gc
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf, open_dict

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import fold_value, register_resolvers

# Before Hydra composes: the configs interpolate ${len:}, ${short_hash:}, ${fold_idx:},
# ${fold_label:} and ${regime_of_teacher:}.
register_resolvers()

from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.checkpoints import (
    Manifest,
    best_or_last_checkpoint,
    checkpoint_of,
    config_hash,
    existing_manifest,
    write_manifest,
)

log = RankedLogger(__name__, rank_zero_only=True)

# The stages fitted on the labels alone; every other stage reads the cell's teacher checkpoint.
STAGES_WITHOUT_TEACHER = ("teacher", "baseline")


def _check_fold(cfg: DictConfig) -> None:
    """:raises ValueError: when ``fold`` is set and outside ``1..data.n_folds``; null
    is the main run."""
    n_folds, fold = int(cfg.data.n_folds), cfg.get("fold")
    if fold is not None and not 1 <= int(fold) <= n_folds:
        raise ValueError(f"fold={fold} is outside 1..{n_folds} (data.n_folds).")


def _wants_hint_checkpoint(cfg: DictConfig) -> bool:
    """True when ``algorithm.warm_start_ckpt`` is the ${hint_ckpt} this function
    fills."""
    raw = OmegaConf.to_container(cfg.algorithm, resolve=False).get("warm_start_ckpt")
    return isinstance(raw, str) and "hint_ckpt" in raw


def _resolve_upstream(cfg: DictConfig) -> None:
    """Turn the upstream run directories into the checkpoint paths the module loads."""
    with open_dict(cfg):
        if cfg.stage not in STAGES_WITHOUT_TEACHER:
            cfg.teacher_ckpt = checkpoint_of(cfg.teacher_run_dir)
        if _wants_hint_checkpoint(cfg):
            cfg.hint_ckpt = checkpoint_of(cfg.hint_run_dir)


def _already_done(cfg: DictConfig, digest: str) -> bool:
    """Whether this run directory already holds a finished run of the same
    configuration.

    :raises FileExistsError: when it holds a run of a different one and
        ``allow_overwrite`` is false, which is a catalogue pointing two hyperparameter
        points at one directory.
    """
    manifest = existing_manifest(cfg.paths.output_dir)
    if manifest is None or (
        cfg.get("allow_overwrite") and manifest.config_hash != digest
    ):
        return False
    if manifest.config_hash == digest:
        log.info(f"Already trained at config_hash {digest}: {cfg.paths.output_dir}")
        return True
    raise FileExistsError(
        f"{cfg.paths.output_dir} holds a run of config_hash {manifest.config_hash}, this one "
        f"is {digest}. Pass allow_overwrite=true to train over it."
    )


def _callbacks_and_loggers(cfg: DictConfig) -> tuple[list[Callback], list[Logger]]:
    log.info("Instantiating callbacks and loggers...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    if not logger:
        # LearningRateMonitor raises, rather than warns, with nothing to log to (the sweeps).
        callbacks = [cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)]
    return callbacks, logger


def _build(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate the datamodule, the module and the trainer of this run."""
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    callbacks, logger = _callbacks_and_loggers(cfg)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    return dict(
        cfg=cfg,
        datamodule=datamodule,
        algorithm=algorithm,
        callbacks=callbacks,
        logger=logger,
        trainer=trainer,
    )


def _mlflow_run_id(trainer: Trainer) -> str | None:
    """The MLflow run this fit logged into, so its evaluation can nest under it."""
    return next(
        (lg.run_id for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None
    )


def _write_manifest(cfg: DictConfig, trainer: Trainer, digest: str) -> None:
    """Record the checkpoint and the run for the stages and the evaluation
    downstream."""
    try:
        checkpoint_path = best_or_last_checkpoint(trainer)
    except RuntimeError as error:
        log.warning(
            f"{error} No manifest written, so nothing downstream can read this run."
        )
        return
    best = trainer.callback_metrics.get(cfg.monitor)
    manifest = Manifest(
        stage=str(cfg.stage),
        fold=fold_value(cfg.fold),
        n_folds=int(cfg.data.n_folds),
        checkpoint_path=checkpoint_path,
        hydra_output_dir=str(cfg.paths.output_dir),
        experiment_name=str(cfg.experiment_name),
        config_hash=digest,
        epochs_run=int(trainer.current_epoch),
        best_metric=None if best is None else float(best),
        mlflow_run_id=_mlflow_run_id(trainer),
    )
    write_manifest(manifest, cfg.paths.output_dir)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the fold ``fold`` of the stage ``stage`` and write its manifest.

    :return: the trainer's callback metrics, empty when the run was already done, and
        the objects the run was built from.
    """
    _check_fold(cfg)
    digest = config_hash(cfg)
    if _already_done(cfg, digest):
        return {}, {}
    _resolve_upstream(cfg)
    # Seeds the initialisation of every (stage, fold) alike; the folds differ by their data.
    L.seed_everything(cfg.seed, workers=True)

    objects = _build(cfg)
    if objects["logger"]:
        log_hyperparameters(objects)

    log.info(f"Starting the {cfg.stage} stage, {cfg.run_label}!")
    trainer: Trainer = objects["trainer"]
    trainer.fit(
        model=objects["algorithm"],
        datamodule=objects["datamodule"],
        ckpt_path=cfg.get("ckpt_path"),
    )
    _write_manifest(cfg, trainer, digest)
    return dict(trainer.callback_metrics), objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    train(cfg)
    # A multirun fits every point in one process, and the trainer, the module and the datamodule
    # reference one another, so a finished point's tensors are cyclic garbage that only a
    # collection frees: CPython's own trigger counts allocations, not bytes.
    gc.collect()


if __name__ == "__main__":
    main()
