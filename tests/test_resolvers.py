from omegaconf import OmegaConf

from src.utils.omegaconf import short_hash


def test_the_fold_is_one_based_and_a_run_name_is_stable() -> None:
    cfg = OmegaConf.create({"fold": 3, "fold_idx": "${fold_idx:${fold}}"})

    assert cfg.fold_idx == 2
    assert short_hash("a/run") == short_hash("a/run") != short_hash("b/run")
    assert short_hash(None)  # --cfg job has no output directory yet
