# Cache one teacher's logits and hints, the inputs every distillation stage trains on.

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${short_hash:} and ${fold_idx:}.
register_resolvers()

from src.algorithms.classifier import Classifier
from src.data.components.teacher_cache import TeacherCache, read_cache, write_cache
from src.metrics.evaluate import net_kwargs, resolve_device
from src.utils import RankedLogger, extras
from src.utils.checkpoints import Manifest, existing_manifest

log = RankedLogger(__name__, rank_zero_only=True)


def cache_teacher(cfg: DictConfig) -> Path:
    """Write the fold's teacher outputs, in the row order of each split, to the cache.

    The distillation stages read the cache rather than a live teacher, so the teacher's
    forward pass is paid once per (dataset, teacher, regime, fold) instead of once per
    epoch of every student.
    """
    datamodule = _datamodule(cfg)
    manifest = _manifest(cfg.teacher_run_dir)
    teacher = _load(manifest.checkpoint_path, datamodule, resolve_device(cfg))
    meta = _meta(cfg, manifest, teacher)
    path = Path(cfg.teacher_cache)
    if _already_cached(path, meta):
        log.info(f"{path} already holds this teacher, so nothing is written.")
        return path
    _write(path, teacher, datamodule, cfg.get("batch_size_eval", 1024), meta)
    return path


def _datamodule(cfg: DictConfig) -> LightningDataModule:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def _manifest(run_dir: str) -> Manifest:
    """The teacher run's manifest.

    :raises FileNotFoundError: when the teacher of this fold has not been trained yet.
    """
    manifest = existing_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"No checkpoint manifest under {run_dir}: train the teacher of this fold first."
        )
    return manifest


def _load(
    checkpoint: str, datamodule: LightningDataModule, device: Any
) -> LightningModule:
    """Load the teacher and bind it to the datamodule's feature layout.

    ``weights_only=False``: the hyperparameters are plain containers, which torch.load
    2.6 refuses to unpickle otherwise.
    """
    log.info(f"Loading the teacher from {checkpoint}")
    teacher = Classifier.load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False
    )
    teacher.bind(datamodule)
    return teacher.eval().to(device)


def _meta(
    cfg: DictConfig, manifest: Manifest, teacher: LightningModule
) -> dict[str, Any]:
    """What a distillation run checks before it trusts the cache it is handed."""
    return {
        "dataset": cfg.data.dataset,
        "teacher": cfg.teacher,
        "regime": cfg.data.regime,
        "fold": cfg.fold,
        "n_folds": cfg.data.n_folds,
        "teacher_ckpt": manifest.checkpoint_path,
        "hint_dim": int(teacher.net.hint_dim),
        "config_hash": manifest.config_hash,
    }


def _already_cached(path: Path, meta: dict[str, Any]) -> bool:
    """Whether *path* already holds this teacher's outputs.

    ``read_cache`` is the only reader of the meta block, so the arrays are read along
    with it; that costs one pass over a file the forward pass below would take minutes
    to rebuild.
    """
    if not path.exists():
        return False
    try:
        read_cache(path, meta)
    except Exception as mismatch:  # any unreadable or stale cache is rebuilt
        log.info(f"Rebuilding {path}: {mismatch}")
        return False
    return True


def _write(
    path: Path,
    teacher: LightningModule,
    datamodule: LightningDataModule,
    batch_size: int,
    meta: dict[str, Any],
) -> None:
    pool = _teacher_outputs(teacher, datamodule, "pool", batch_size)
    test = _teacher_outputs(teacher, datamodule, "test", batch_size)
    write_cache(path, TeacherCache(*pool, *test, meta))
    log.info(f"Wrote {path}: pool {tuple(pool[0].shape)}, test {tuple(test[0].shape)}")


@torch.no_grad()
def _teacher_outputs(
    teacher: LightningModule,
    datamodule: LightningDataModule,
    split: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The teacher's logits and hints over one split, in that split's own row order."""
    device = next(teacher.parameters()).device
    logits: list[torch.Tensor] = []
    hints: list[torch.Tensor] = []
    # Through the module rather than the network: it reduces the batch to the columns the
    # network reads and de-normalises them first.
    for x, _, matrices in datamodule.eval_batches(split, batch_size):
        moved = None if matrices is None else matrices.to(device).float()
        kwargs = net_kwargs(teacher, moved)
        batch = teacher.forward_with_hint(x.to(device).float(), **kwargs)
        logits.append(batch[0].cpu())
        hints.append(batch[1].cpu())
    return torch.cat(logits).numpy(), torch.cat(hints).numpy()


@hydra.main(version_base="1.3", config_path="../configs", config_name="cache.yaml")
def main(cfg: DictConfig) -> None:
    """Cache the teacher outputs ``cfg`` names."""
    extras(cfg)
    cache_teacher(cfg)


if __name__ == "__main__":
    main()
