from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.train import _wants_hint_checkpoint
from tests.conftest import EXPERIMENTS, TRAINING_STAGES, compose_root, point_into

WARMUP_COSINE_KEYS = {
    "_target_",
    "_partial_",
    "warmup_epochs",
    "max_epochs",
    "min_lr_ratio",
}


def _composed(tmp_path: Path, *overrides: str) -> DictConfig:
    cfg = compose_root("train.yaml", *overrides, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))
    return cfg


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("method", TRAINING_STAGES)
def test_every_experiment_and_method_composes_into_its_own_run_directory(
    experiment: str, method: str, tmp_path: Path
) -> None:
    cfg = _composed(tmp_path, f"experiment={experiment}", f"method={method}")

    assert cfg.stage == method and cfg.run_dir.endswith(cfg.hp_tag)
    assert f"fold_{cfg.fold}" in cfg.run_dir and cfg.data.regime in cfg.run_dir
    # A multirun point lands where the single run of that point would.
    assert (
        cfg.hydra.run.dir.endswith(cfg.run_dir) and cfg.hydra.sweep.subdir == cfg.hp_tag
    )
    assert cfg.trainer.max_epochs >= 200


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_the_data_and_the_algorithm_of_every_experiment_build(
    experiment: str, tmp_path: Path
) -> None:
    cfg = _composed(tmp_path, f"experiment={experiment}", "method=kd")

    datamodule = hydra.utils.instantiate(cfg.data)
    algorithm = hydra.utils.instantiate(cfg.algorithm)

    assert (datamodule.regime, datamodule.n_folds, datamodule.fold_idx) == (
        "canonical",
        5,
        0,
    )
    assert list(algorithm.hparams.input_shape) == list(cfg.input_shape)


def test_the_stages_carry_their_own_recipes(tmp_path: Path) -> None:
    hint = _composed(tmp_path, "experiment=jets_deepsets", "method=hint")
    kd_hint = _composed(tmp_path, "experiment=jets_deepsets", "method=kd_hint")
    relational = _composed(tmp_path, "experiment=jets_lorentznet", "method=relational")
    teacher = _composed(tmp_path, "experiment=jets_lorentznet", "method=teacher")
    hot = _composed(
        tmp_path, "experiment=jets_deepsets", "method=kd", "algorithm.temperature=16"
    )

    assert (hint.monitor, hint.monitor_mode, hint.algorithm.freeze_above_guided) == (
        "val/hint_mse",
        "min",
        True,
    )
    assert hint.data.teacher_cache == hint.teacher_cache
    assert _wants_hint_checkpoint(kd_hint) and not _wants_hint_checkpoint(hint)
    assert (
        set(relational.algorithm.scheduler.scheduler)
        == set(teacher.algorithm.scheduler.scheduler)
        == WARMUP_COSINE_KEYS
    )
    assert (relational.trainer.max_epochs, relational.data.batch_size) == (
        300,
        relational.relational_batch_size,
    )
    assert (teacher.algorithm.optimizer._target_, teacher.data.batch_size) == (
        "torch.optim.AdamW",
        teacher.teacher_batch_size,
    )
    assert (
        hot.run_dir
        != _composed(tmp_path, "experiment=jets_deepsets", "method=kd").run_dir
    )


def test_the_evaluation_cache_and_debug_configs_compose(tmp_path: Path) -> None:
    evaluation = compose_root(
        "eval.yaml", "experiment=mnist_cnn", "method=kd", "trainer=cpu"
    )
    cache = compose_root(
        "cache.yaml", "experiment=mnist_cnn", "method=cache", "trainer=cpu"
    )
    debug = compose_root(
        "train.yaml", "experiment=mnist_cnn", "method=kd", "debug=limit"
    )
    HydraConfig().set_config(point_into(evaluation, tmp_path))

    assert evaluation.checkpoint_manifest_path.endswith("checkpoint_manifest.json")
    assert (
        list(evaluation.eval_splits) == ["fold", "test"]
        and evaluation.probe.n_draws == 10
    )
    assert cache.stage == "cache" and cache.data.batch_size == cache.teacher_batch_size
    assert (
        debug.task_name,
        debug.train_task,
        debug.results.path,
        debug.data.n_folds,
    ) == ("debug", "debug", None, 2)
