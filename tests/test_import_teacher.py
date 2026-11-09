import hashlib
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.algorithms.classifier import Classifier
from src.algorithms.components.deepsets import DeepSetsNet
from src.import_teacher import import_teacher, map_state_dict
from src.train import train
from src.utils.checkpoints import config_hash, existing_manifest
from tests.conftest import compose_root, point_into

COMMON = ("experiment=jets_deepsets", "trainer=cpu", "logger=none", "fold=1")


def _sibling_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    # The layout of a jet_tagging_gdl checkpoint: the network under net., the best score
    # in the ModelCheckpoint state.
    state = {f"net.{k}": v for k, v in DeepSetsNet().state_dict().items()}
    best = {"best_model_score": torch.tensor(0.8)}
    callbacks = {"ModelCheckpoint{'monitor': 'val/acc', 'mode': 'max'}": best}
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": state, "epoch": 7, "callbacks": callbacks}, path)
    return state


def _configs(tmp_path: Path, source: Path) -> tuple[DictConfig, DictConfig]:
    imported = compose_root("import_teacher.yaml", *COMMON, f"source_ckpt={source}")
    trained = compose_root("train.yaml", *COMMON, "method=teacher")
    return point_into(imported, tmp_path), point_into(trained, tmp_path)


def test_an_imported_teacher_loads_and_train_would_not_retrain_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sibling" / "epoch_007.ckpt"
    state = _sibling_checkpoint(source)
    imported, trained = _configs(tmp_path / "run", source)
    HydraConfig().set_config(imported)

    import_teacher(imported)
    manifest = existing_manifest(tmp_path / "run")
    module = Classifier.load_from_checkpoint(
        manifest.checkpoint_path, weights_only=False
    )

    assert imported.hydra.run.dir == trained.hydra.run.dir
    assert manifest.config_hash == config_hash(trained)
    assert (manifest.stage, manifest.fold, manifest.epochs_run) == ("teacher", 1, 8)
    assert manifest.best_metric == pytest.approx(0.8)
    assert manifest.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(torch.equal(module.state_dict()[k], v) for k, v in state.items())
    # Composed afresh: instantiating the module resolved the first config in place.
    again, trained = _configs(tmp_path / "run", source)
    assert import_teacher(again) == ({}, {})
    assert train(trained) == ({}, {})


def test_the_key_map_drops_and_then_renames() -> None:
    state = {"denorm_scales": 1, "net.a.weight": 2, "net.b": 3}

    mapped = map_state_dict(state, ["denorm_scales"], {"net.a.": "net.graph.a."})

    assert mapped == {"net.graph.a.weight": 2, "net.b": 3}
