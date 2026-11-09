from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.cache import cache_teacher
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
    assert f"/{cfg.run_label}/" in cfg.run_dir and cfg.data.regime in cfg.run_dir
    # A multirun point lands where the single run of that point would.
    assert (
        cfg.hydra.run.dir.endswith(cfg.run_dir) and cfg.hydra.sweep.subdir == cfg.hp_tag
    )
    assert cfg.trainer.max_epochs >= 200


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_the_data_and_the_algorithm_of_every_experiment_build(
    experiment: str, tmp_path: Path
) -> None:
    cfg = _composed(tmp_path, f"experiment={experiment}", "method=kd", "fold=1")

    datamodule = hydra.utils.instantiate(cfg.data)
    algorithm = hydra.utils.instantiate(cfg.algorithm)

    assert (datamodule.regime, datamodule.n_folds, datamodule.fold_idx) == (
        "canonical",
        5,
        0,
    )
    assert list(algorithm.hparams.input_shape) == list(cfg.input_shape)


def test_the_main_run_is_the_default_and_lands_under_main(tmp_path: Path) -> None:
    train = _composed(tmp_path, "experiment=jets_deepsets", "method=kd")
    cache = compose_root(
        "cache.yaml", "experiment=jets_deepsets", "method=cache", "trainer=cpu"
    )
    evaluation = compose_root(
        "eval.yaml", "experiment=jets_deepsets", "method=kd", "trainer=cpu"
    )
    fold = _composed(tmp_path, "experiment=jets_deepsets", "method=kd", "fold=2")

    assert train.run_dir.endswith(f"/kd/main/{train.hp_tag}")
    assert "/teacher/main/" in train.teacher_run_dir
    assert "/hint/main/" in train.hint_run_dir
    assert train.data.teacher_cache.endswith("/canonical/main.npz")
    assert cache.teacher_cache.endswith("/main.npz")
    assert "/teacher/main/" in cache.teacher_run_dir
    assert evaluation.checkpoint_manifest_path.endswith(
        f"/kd/main/{evaluation.hp_tag}/checkpoint_manifest.json"
    )
    assert train.data.fold_idx is None and cache.data.fold_idx is None
    assert evaluation.data.fold_idx is None
    assert fold.data.fold_idx == 1 and fold.teacher_cache.endswith("/fold_2.npz")


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
    assert set(relational.algorithm.scheduler.scheduler) == WARMUP_COSINE_KEYS
    assert teacher.algorithm.scheduler.scheduler.patience == 80
    assert (relational.trainer.max_epochs, relational.data.batch_size) == (
        300,
        relational.relational_batch_size,
    )
    assert (teacher.algorithm.optimizer._target_, teacher.data.batch_size) == (
        "torch.optim.Adam",
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
    assert (evaluation.eval_data, evaluation.probe_subgroups) == (None, None)
    assert evaluation.dump_samples == "cell"
    assert cache.stage == "cache" and cache.data.batch_size == cache.teacher_batch_size
    assert (
        debug.task_name,
        debug.train_task,
        debug.results.path,
        debug.data.n_folds,
    ) == ("debug", "debug", None, 2)
    assert debug.teacher_cache.endswith(
        "/debug/teacher_cache/mnist/cnn_circular/canonical/main.npz"
    )


def test_the_augmented_regime_distils_the_canonical_teacher(tmp_path: Path) -> None:
    augmented = (
        "experiment=jets_deepsets",
        "method=kd",
        "fold=1",
        "data.regime=augmented",
    )
    kd = _composed(tmp_path, *augmented)
    debug = compose_root("train.yaml", *augmented, "debug=limit")

    assert "/canonical/teacher/fold_1/" in kd.teacher_run_dir
    assert kd.teacher_cache.endswith("/canonical/fold_1.npz")
    assert kd.data.teacher_cache == kd.teacher_cache
    assert "/augmented_permutation/hint/fold_1/" in kd.hint_run_dir
    assert kd.run_dir.startswith("jets/deepsets/augmented_permutation/kd/fold_1/")
    assert debug.teacher_cache.endswith(
        "/debug/teacher_cache/jets/deepsets/canonical/fold_1.npz"
    )


def test_a_transformed_student_reads_the_canonical_teacher_and_its_own_cache(
    tmp_path: Path,
) -> None:
    kd = _composed(
        tmp_path,
        "experiment=jets_jetformer",
        "method=kd",
        "data.regime=transformed",
        "data.group=so3",
    )

    assert "/jetformer/canonical/teacher/main/" in kd.teacher_run_dir
    assert kd.teacher_cache.endswith("/jetformer/transformed_so3/main.npz")
    assert kd.run_dir.startswith("jets/jetformer/transformed_so3/kd/main/")


def test_the_seed_names_every_student_run_and_its_hint_run(tmp_path: Path) -> None:
    runs = {
        seed: _composed(
            tmp_path, "experiment=jets_deepsets", "method=kd_hint", f"seed={seed}"
        )
        for seed in (12345, 12346)
    }
    teacher = _composed(tmp_path, "experiment=jets_deepsets", "method=teacher")

    assert runs[12345].hp_tag.endswith("_s12345")
    assert runs[12345].run_dir != runs[12346].run_dir
    assert runs[12346].hint_run_dir.endswith("/hint/main/lr0.001_s12346")
    assert runs[12345].teacher_run_dir == runs[12346].teacher_run_dir
    assert "_s" not in teacher.hp_tag


@pytest.mark.parametrize(
    "config", ["eval.yaml", "orbit_profile.yaml", "concept_probe.yaml"]
)
def test_a_seeded_checkpoint_is_probed_on_the_draws_of_every_other(config: str) -> None:
    cfg = compose_root(
        config, "experiment=jets_deepsets", "method=kd", "trainer=cpu", "seed=12346"
    )

    assert cfg.hp_tag.endswith("_s12346") and cfg.probe_seed == 12345


@pytest.mark.parametrize(
    ("experiment", "features"),
    [
        ("jets_deepsets", ["pt", "etarel", "phirel"]),
        ("jets_mlp", ["pt", "etarel", "phirel"]),
        ("jets_lorentznet", ["pt", "eta", "phi"]),
        ("jets_lloca_z", ["pt", "eta", "phi"]),
        ("jets_lloca_so3", ["pt", "eta", "phi"]),
        ("jets_jetformer", ["pt", "eta", "phi"]),
        ("jets_jetformer_symm_so2", ["pt", "eta", "phi"]),
    ],
)
def test_every_jets_cell_trains_its_students_alike(
    experiment: str, features: list[str], tmp_path: Path
) -> None:
    kd = _composed(tmp_path, f"experiment={experiment}", "method=kd")
    relational = _composed(tmp_path, f"experiment={experiment}", "method=relational")

    assert list(kd.student_net.required_features) == features
    assert (kd.data.batch_size, relational.data.batch_size) == (1024, 1024)
    assert kd.trainer.gradient_clip_val is None


def test_the_acquisition_callbacks_add_a_periodic_checkpoint_after_the_best(
    tmp_path: Path,
) -> None:
    cfg = _composed(
        tmp_path, "experiment=jets_deepsets", "method=kd", "callbacks=acquisition"
    )
    names = list(cfg.callbacks)

    assert names.index("model_checkpoint") < names.index("acquisition_checkpoint")
    assert cfg.callbacks.acquisition_checkpoint.every_n_epochs == 5
    assert cfg.callbacks.acquisition_checkpoint.save_top_k == -1


def test_the_cache_refuses_the_augmented_regime_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = compose_root(
        "cache.yaml",
        "experiment=mnist_cnn",
        "method=cache",
        "trainer=cpu",
        "fold=1",
        "data.regime=augmented",
    )
    monkeypatch.setattr("src.cache._datamodule", lambda _: pytest.fail("loaded data"))

    with pytest.raises(ValueError, match="data.regime=canonical"):
        cache_teacher(cfg)
