from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from src.utils.checkpoints import config_hash, existing_manifest
from src.vocabulary import MAIN_FOLD
from tests.conftest import EXPERIMENTS, compose_root, needs_record, point_into

pytestmark = needs_record


def one_batch(experiment: str, tmp_path: Path, *overrides: str) -> DictConfig:
    cfg = compose_root(
        "train.yaml",
        f"experiment={experiment}",
        "method=teacher",
        "trainer=cpu",
        "logger=none",
        *overrides,
    )
    HydraConfig().set_config(point_into(cfg, tmp_path))
    with open_dict(cfg):
        (
            cfg.data.n_folds,
            cfg.data.num_workers,
            cfg.data.pin_memory,
            cfg.data.batch_size,
        ) = (2, 0, False, 32)
        cfg.trainer.max_epochs = (
            cfg.trainer.limit_train_batches
        ) = cfg.trainer.limit_val_batches = 1
        cfg.extras.print_config = cfg.extras.enforce_tags = False
    return cfg


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_one_batch_of_a_teacher_leaves_a_manifest_of_its_checkpoint(
    experiment: str, tmp_path: Path
) -> None:
    cfg = one_batch(experiment, tmp_path)
    digest = config_hash(
        cfg
    )  # before the fit: instantiate resolves the config in place

    metrics, _ = train(cfg)
    manifest = existing_manifest(tmp_path)

    assert (
        Path(manifest.checkpoint_path).is_absolute()
        and Path(manifest.checkpoint_path).exists()
    )
    assert (
        manifest.stage,
        manifest.fold,
        manifest.n_folds,
        manifest.epochs_run,
        manifest.config_hash,
    ) == ("teacher", MAIN_FOLD, 2, 1, digest)
    assert float(metrics[cfg.monitor]) == pytest.approx(manifest.best_metric)


def test_a_finished_run_is_neither_repeated_nor_silently_overwritten(
    tmp_path: Path,
) -> None:
    train(one_batch("mnist_cnn", tmp_path))

    assert train(one_batch("mnist_cnn", tmp_path)) == ({}, {})
    with pytest.raises(FileExistsError, match="allow_overwrite"):
        train(one_batch("mnist_cnn", tmp_path, "seed=1"))
    with pytest.raises(ValueError, match="fold="):
        train(one_batch("mnist_cnn", tmp_path, "fold=3"))


def test_a_fold_run_records_its_fold_as_an_integer(tmp_path: Path) -> None:
    train(one_batch("mnist_cnn", tmp_path, "fold=1"))

    assert existing_manifest(tmp_path).fold == 1
