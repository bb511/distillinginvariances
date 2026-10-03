import os
from pathlib import Path

import pytest
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.components.folds import FoldedDataModule
from src.utils.omegaconf import register_resolvers
from src.vocabulary import STAGES

register_resolvers()

ROOT = rootutils.find_root(indicator=".project-root")
# configs/paths/default.yaml reads it, and a bare pytest has neither .env nor the variable.
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
CACHE_DIR = os.environ.get("HLS4ML_CACHE_DIR", str(ROOT / "data" / "hf"))

# Read here: importing src.train loads .env, after which the gate would never skip.
needs_record = pytest.mark.skipif(
    "HF_HUB_CACHE" not in os.environ, reason="HF_HUB_CACHE names the hub cache."
)

EXPERIMENTS = ("mnist_cnn", "mnist_cnn_zeros", "jets_deepsets", "jets_lorentznet")
TRAINING_STAGES = tuple(stage for stage in STAGES if stage != "cache")
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
FOUR_MOMENTUM = ["px", "py", "pz", "e"]
# GeV. A massless constituent puts m^2 at the float32 rounding of E^2 - |p|^2, and a boost
# test would then measure the arithmetic.
MASS = 20.0


def jets(n: int = 6, slots: int = 8, real: int = 5, columns: int = 4, seed: int = 0):
    """(jets, slots, columns) with [px, py, pz, e] first, padded slots all-zero."""
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n, slots, 3, generator=generator) * MASS
    batch = torch.zeros(n, slots, columns)
    batch[..., :3] = momenta
    batch[..., 3] = torch.sqrt((momenta**2).sum(-1) + MASS**2)
    batch[:, real:] = 0.0
    return batch


class TinyJets(FoldedDataModule):
    """40 synthetic jets in three classes; the main run validates on the last ten."""

    num_classes = 3
    input_shape = (8, 4)
    nconstituents = 8
    feature_names = FOUR_MOMENTUM

    def load_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        return jets(40, seed=1), torch.eye(3)[torch.arange(40) % 3]

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return jets(8, seed=2), torch.eye(3)[torch.arange(8) % 3]

    def main_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.arange(30), torch.arange(30, 40)


TINY_JETS = {
    "dataset": "tiny",
    "regime": "canonical",
    "group": "lorentz_iso",
    "transform_seed": 7,
    "transform_kwargs": {"lorentz_eta_max": 1.0},
    "batch_size": 8,
    "num_workers": 0,
    "pin_memory": False,
    "n_folds": 2,
    "kfold_seed": 42,
    "fold_idx": 0,
    "teacher_cache": None,
}


def tiny_jets(tmp_path: Path, **overrides) -> TinyJets:
    settings = TINY_JETS | {"transformed_dir": str(tmp_path / "transformed")}
    datamodule = TinyJets(**(settings | overrides))
    datamodule.setup()
    return datamodule


def compose_root(config_name: str, *overrides: str) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name, return_hydra_config=True, overrides=list(overrides))
    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT)
    return cfg


def point_into(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    # set_config makes the hydra node read-only, and run_name needs a live output_dir.
    OmegaConf.set_readonly(cfg.hydra, False)
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path / "results")
        cfg.hydra.runtime.output_dir = str(tmp_path)
    return cfg
