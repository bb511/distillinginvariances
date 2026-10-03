from omegaconf import OmegaConf

from src.utils.omegaconf import fold_value, short_hash
from src.vocabulary import MAIN_FOLD


def test_the_fold_is_one_based_and_a_run_name_is_stable() -> None:
    cfg = OmegaConf.create({"fold": 3, "fold_idx": "${fold_idx:${fold}}"})

    assert cfg.fold_idx == 2
    assert short_hash("a/run") == short_hash("a/run") != short_hash("b/run")
    assert short_hash(None)  # --cfg job has no output directory yet


def test_a_null_fold_is_the_main_run() -> None:
    cfg = OmegaConf.create(
        {
            "fold": None,
            "fold_idx": "${fold_idx:${fold}}",
            "run_label": "${fold_label:${fold},main}",
        }
    )

    assert (cfg.fold_idx, cfg.run_label) == (None, "main")
    assert fold_value(cfg.fold) == MAIN_FOLD
    cfg.fold = 2
    assert (cfg.fold_idx, cfg.run_label, fold_value(cfg.fold)) == (1, "fold_2", 2)


def test_the_augmented_regime_reads_the_canonical_teacher() -> None:
    cfg = OmegaConf.create(
        {"regime": "augmented", "teacher_regime": "${regime_of_teacher:${regime}}"}
    )

    assert cfg.teacher_regime == "canonical"
    cfg.regime = "transformed"
    assert cfg.teacher_regime == "transformed"
