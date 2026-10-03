from pathlib import Path

from src.utils.checkpoints import (
    Manifest,
    checkpoint_of,
    config_hash,
    existing_manifest,
    write_manifest,
)
from src.vocabulary import MAIN_FOLD
from tests.conftest import compose_root


def test_a_manifest_round_trips(tmp_path: Path) -> None:
    manifest = Manifest("kd", 2, 5, "/x/epoch_003.ckpt", str(tmp_path), "t", "abc", 4)

    assert existing_manifest(tmp_path) is None
    write_manifest(manifest, tmp_path)

    assert existing_manifest(tmp_path) == manifest
    assert checkpoint_of(tmp_path) == "/x/epoch_003.ckpt"


def test_a_manifest_of_the_main_run_round_trips(tmp_path: Path) -> None:
    manifest = Manifest("kd", MAIN_FOLD, 5, "/x/last.ckpt", str(tmp_path), "t", "a", 4)

    write_manifest(manifest, tmp_path)

    assert existing_manifest(tmp_path).fold == MAIN_FOLD
    assert existing_manifest(tmp_path) == manifest


def test_the_hash_follows_the_science_and_not_the_machine() -> None:
    base = compose_root("train.yaml", "experiment=jets_deepsets", "method=kd")
    card = compose_root(
        "train.yaml", "experiment=jets_deepsets", "method=kd", "trainer.devices=[1]"
    )
    hotter = compose_root(
        "train.yaml",
        "experiment=jets_deepsets",
        "method=kd",
        "algorithm.temperature=16",
    )
    teacher = compose_root("train.yaml", "experiment=jets_deepsets", "method=teacher")
    slower = compose_root(
        "train.yaml", "experiment=jets_deepsets", "method=teacher", "teacher_lr=1e-4"
    )

    assert config_hash(base) == config_hash(card)
    assert config_hash(base) != config_hash(hotter)
    assert config_hash(teacher) != config_hash(slower)
