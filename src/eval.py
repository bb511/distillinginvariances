# Evaluate one trained checkpoint and append its rows to the results table.

from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${short_hash:} and ${fold_idx:}.
register_resolvers()

from src import vocabulary
from src.algorithms.classifier import Classifier
from src.algorithms.distillation import DistillationModule
from src.metrics.evaluate import constant_row, evaluate_checkpoint, resolve_device
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.checkpoints import Manifest, checkpoint_of, read_manifest
from src.utils.results import ResultsWriter

log = RankedLogger(__name__, rank_zero_only=True)

# The stages whose checkpoint holds a plain classifier; every other stage distils.
CLASSIFIER_STAGES = ("teacher", "baseline")
# The hyperparameter columns each stage's objective reads. The rest stay empty, so a row is
# keyed by the knobs that produced it and no column carries a value nothing acted on.
STAGE_KNOBS = {
    "kd": ("temperature", "alpha", "beta"),
    "kd_hint": ("temperature", "alpha", "beta"),
    "hint_beta": ("temperature", "alpha", "beta"),
    "relational": ("alpha", "beta", "discrepancy", "lambda_is", "lambda_ic"),
}
# Written blank on the constant-predictor row: it is fitted by nothing and trained on nothing.
CONSTANT_BLANKS = (
    "temperature",
    "alpha",
    "beta",
    "discrepancy",
    "lambda_is",
    "lambda_ic",
    "lr",
    "batch_size",
    "epochs_run",
    "guided_index",
    "checkpoint_path",
)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one checkpoint on every split of ``cfg.eval_splits``.

    The validation fold is evaluated first and its temperature scales the test logits,
    so the test row's calibration after scaling is out of sample.

    :return: every split's metrics, keyed ``<split>/<metric>``, and the shared objects.
    """
    if cfg.stage not in vocabulary.METHOD_OF_STAGE:
        raise ValueError(
            f"The {cfg.stage} stage is never exported; nothing to evaluate."
        )
    manifest, checkpoint = _checkpoint(cfg)
    datamodule = _datamodule(cfg)
    device = resolve_device(cfg)
    module = _load(cfg.stage, checkpoint, datamodule, device)
    teacher = module if cfg.stage == "teacher" else _teacher(cfg, datamodule, device)
    metrics = _evaluate_splits(cfg, module, teacher, datamodule)
    _write(cfg, _rows(cfg, manifest, checkpoint, datamodule, metrics))
    return _flat(metrics), {"cfg": cfg, "datamodule": datamodule, "module": module}


def _checkpoint(cfg: DictConfig) -> tuple[Manifest | None, str]:
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


def _datamodule(cfg: DictConfig) -> LightningDataModule:
    """The datamodule, set up here: the evaluation pass runs without a Trainer."""
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def _load(
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


def _teacher(
    cfg: DictConfig, datamodule: LightningDataModule, device: Any
) -> LightningModule:
    """The teacher the fidelity columns are measured against, the baseline's
    included."""
    return _load("teacher", checkpoint_of(cfg.teacher_run_dir), datamodule, device)


def _evaluate_splits(
    cfg: DictConfig,
    module: LightningModule,
    teacher: LightningModule,
    datamodule: LightningDataModule,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    temperature = None
    for split in _ordered_splits(cfg):
        # Re-seeded per split: the probe draws from the global RNG, so two checkpoints
        # compared with each other must meet the same transforms.
        L.seed_everything(cfg.seed, workers=True)
        log.info(f"Evaluating the {split!r} split")
        results[split] = evaluate_checkpoint(
            module,
            teacher,
            datamodule,
            split,
            cfg.probe,
            temperature,
            cfg.get("batch_size_eval", 1024),
        )
        if split == "fold":
            temperature = results[split]["temperature_ts"]
    return results


def _ordered_splits(cfg: DictConfig) -> list[str]:
    splits = list(cfg.get("eval_splits") or vocabulary.EVAL_SPLITS)
    # The fold's temperature scales the test logits, so the fold comes first.
    return sorted(splits, key=lambda split: split != "fold")


def _rows(
    cfg: DictConfig,
    manifest: Manifest | None,
    checkpoint: str,
    datamodule: LightningDataModule,
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per split, and the constant-predictor floor beside the teacher's rows."""
    base = _identity(cfg) | _provenance(cfg, manifest, checkpoint)
    rows = [
        base | {"eval_split": split} | _metric_cells(values)
        for split, values in metrics.items()
    ]
    if cfg.stage != "teacher":
        return rows
    return rows + [_constant(cfg, base, datamodule, split) for split in metrics]


def _identity(cfg: DictConfig) -> dict[str, Any]:
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
        "fold": cfg.fold,
        "n_folds": cfg.data.n_folds,
        "group": cfg.data.group,
    } | knobs


def _provenance(
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


def _constant(
    cfg: DictConfig, base: dict[str, Any], datamodule: LightningDataModule, split: str
) -> dict[str, Any]:
    """The class-frequency floor, written once per split under the teacher stage."""
    metrics = constant_row(datamodule, split, cfg.get("batch_size_eval", 1024))
    labels = {"method": "constant", "stage": "constant", "eval_split": split}
    return base | dict.fromkeys(CONSTANT_BLANKS, "") | labels | _metric_cells(metrics)


def _metric_cells(metrics: dict[str, Any]) -> dict[str, Any]:
    """The metric columns alone: the logits and the targets stay out of the table."""
    return {
        name: value
        for name, value in metrics.items()
        if name not in ("logits", "targets")
    }


def _write(cfg: DictConfig, rows: list[dict[str, Any]]) -> None:
    """Append the rows, unless this is a throwaway evaluation (``results.path:

    null``).
    """
    path = OmegaConf.select(cfg, "results.path")
    if path is None:
        log.info("results.path is null, so the evaluation writes no row.")
        return
    writer = ResultsWriter(path)
    for row in rows:
        writer.append(row)
    log.info(f"Wrote {len(rows)} rows to {path}")


def _flat(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        f"{split}/{name}": value
        for split, values in metrics.items()
        for name, value in _metric_cells(values).items()
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoint ``cfg`` points at."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
