"""What the evaluating entry points (eval, coverage, profile, probe) share: the
checkpoint to read, the rows to serve, the module to load and the identity its rows
carry."""

from typing import Any

import hydra
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

from src import vocabulary
from src.algorithms.classifier import Classifier
from src.algorithms.distillation import DistillationModule
from src.metrics.evaluate import Probe, make_probe
from src.utils.checkpoints import Manifest, checkpoint_of, read_manifest
from src.utils.omegaconf import fold_value
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# The stages whose checkpoint holds a plain classifier; every other stage distils.
CLASSIFIER_STAGES = ("teacher", "baseline", "label_smoothing")
# The hyperparameter columns each stage's objective reads. The rest stay empty, so a row is
# keyed by the knobs that produced it and no column carries a value nothing acted on.
STAGE_KNOBS = {
    "kd": ("temperature", "alpha", "beta"),
    "kd_hint": ("temperature", "alpha", "beta"),
    "hint_beta": ("temperature", "alpha", "beta"),
    "relational": ("alpha", "beta", "discrepancy", "lambda_is", "lambda_ic"),
    "logit_matching": ("alpha",),
    "label_smoothing": ("label_smoothing",),
}


def resolve_checkpoint(cfg: DictConfig) -> tuple[Manifest | None, str]:
    """The one checkpoint to evaluate, named by its manifest or outright.

    :raises ValueError: when neither is set, so a mistyped path fails rather than
        evaluating nothing.
    """
    path = cfg.get("checkpoint_manifest_path")
    if path:
        manifest = read_manifest(path)
        return manifest, manifest.checkpoint_path
    if cfg.get("ckpt_path"):
        return None, str(cfg.ckpt_path)
    raise ValueError(
        "Nothing to evaluate: set checkpoint_manifest_path=... or ckpt_path=..."
    )


def eval_data(cfg: DictConfig) -> str:
    """The regime whose rows are evaluated: ``eval_data``, or the run's own."""
    rows = cfg.get("eval_data") or vocabulary.DATA_OF_REGIME[cfg.data.regime]
    if rows not in vocabulary.EVAL_DATA:
        raise ValueError(
            f"eval_data must be one of {list(vocabulary.EVAL_DATA)}, got {rows!r}."
        )
    return rows


def eval_datamodule(cfg: DictConfig) -> LightningDataModule:
    """The datamodule serving the eval_data rows, set up here: no Trainer runs the pass.

    It reads no teacher cache. The pass computes the teacher's outputs itself, and the
    cache's regime check would refuse the rows of another regime.
    """
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, regime=eval_data(cfg), teacher_cache=None
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def probes_of(cfg: DictConfig) -> list[Probe]:
    """The cell's own group first, then ``probe_subgroups`` (null: the dataset's)."""
    subgroups = cfg.get("probe_subgroups")
    if subgroups is None:
        subgroups = vocabulary.PROBE_SUBGROUPS[cfg.data.dataset]
    groups = dict.fromkeys([cfg.data.group, *subgroups])
    return [make_probe(group, cfg.probe) for group in groups]


def load_module(
    stage: str, checkpoint: str, datamodule: LightningDataModule, device: Any
) -> LightningModule:
    """Load the module of *stage* and bind it to the datamodule's feature layout.

    ``weights_only=False``: the hyperparameters are plain containers, which torch.load
    2.6 refuses to unpickle otherwise.
    """
    module_class = Classifier if stage in CLASSIFIER_STAGES else DistillationModule
    log.info(f"Loading {module_class.__name__} from {checkpoint}")
    module = module_class.load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False
    )
    module.bind(datamodule)
    return module.eval().to(device)


def load_teacher(
    cfg: DictConfig, datamodule: LightningDataModule, device: Any
) -> LightningModule:
    """The teacher the fidelity columns are measured against, the baseline's
    included."""
    checkpoint = checkpoint_of(cfg.teacher_run_dir)
    return load_module("teacher", checkpoint, datamodule, device)


def row_identity(cfg: DictConfig) -> dict[str, Any]:
    """What was trained: the axes of the study and the knobs of this stage's
    objective."""
    knobs = {name: cfg.algorithm.get(name) for name in STAGE_KNOBS.get(cfg.stage, ())}
    return {
        "dataset": cfg.data.dataset,
        "teacher": cfg.teacher,
        "regime": cfg.data.regime,
        "method": vocabulary.METHOD_OF_STAGE[cfg.stage],
        "lr": OmegaConf.select(cfg, "algorithm.optimizer.lr"),
        "batch_size": cfg.data.batch_size,
        "seed": cfg.seed,
        "fold": fold_value(cfg.fold),
        "n_folds": cfg.data.n_folds,
        "group": cfg.data.group,
    } | knobs


def row_provenance(
    cfg: DictConfig, manifest: Manifest | None, checkpoint: str
) -> dict[str, Any]:
    """Where the row came from; a bare ``ckpt_path`` carries no manifest, hence the
    blanks."""
    return {
        "stage": cfg.stage,
        "variant": cfg.get("variant") or "",
        "optimizer": _target_name(cfg, "algorithm.optimizer"),
        "scheduler": _target_name(cfg, "algorithm.scheduler.scheduler"),
        "guided_index": cfg.algorithm.get("guided_index"),
        "checkpoint_path": checkpoint,
        "run_dir": manifest.hydra_output_dir if manifest else "",
        "epochs_run": manifest.epochs_run if manifest else 0,
        "config_hash": manifest.config_hash if manifest else "",
    }


def _target_name(cfg: DictConfig, key: str) -> str:
    """The class of a ``_target_``, without its module path; empty when there is
    none."""
    return str(OmegaConf.select(cfg, f"{key}._target_") or "").rsplit(".", 1)[-1]


def reinitialise(module: LightningModule, seed: int) -> None:
    """The untrained control: every layer of the network at a seeded initialisation."""
    torch.manual_seed(seed)
    for layer in module.net.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
