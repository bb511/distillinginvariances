import os
from pathlib import Path

import numpy as np
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

EXPERIMENTS = (
    "mnist_cnn",
    "mnist_cnn_zeros",
    "jets_deepsets",
    "jets_lorentznet",
    "jets_lorentznet_m2",
    "jets_mlp",
    "jets_lloca_z",
    "jets_lloca_so3",
    "jets_lloca_lorentz",
    "jets_jetformer",
    "jets_jetformer_aug_so2",
    "jets_jetformer_symm_so2",
)
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


# Jets per raw file of the synthetic top-tagging record, converted in chunks of 16: the
# training file in three reads, the last one partial. N = 8 truncates the jets of more
# constituents.
TOP_TAGGING_SIZES = {"train.h5": 40, "val.h5": 16, "test.h5": 12}
TOP_TAGGING_N = 8
# The jets of every synthetic file whose first two constituents are swapped.
TOP_TAGGING_PLANTED = (1, 3)


def top_tagging_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    # The raw file *name* as the record stores it: [E, px, py, pz] in GeV (jets, 200, 4)
    # of 1 to 20 massless constituents at p_T U(1, 100) GeV in falling p_T but in the
    # planted jets, zero-padded, and is_signal_new alternating from 0.
    n_jets = TOP_TAGGING_SIZES[name]
    rng = np.random.default_rng(list(TOP_TAGGING_SIZES).index(name))
    pt = -np.sort(-rng.uniform(1.0, 100.0, (n_jets, 20)), axis=1)
    eta, phi = (rng.normal(centre, 0.3, (n_jets, 20)) for centre in (0.5, 0.2))
    p4 = np.stack(
        [pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1
    )
    multiplicity = rng.integers(1, 21, n_jets)
    multiplicity[list(TOP_TAGGING_PLANTED)] = 20
    p4[np.arange(20) >= multiplicity[:, None]] = 0.0
    p4[list(TOP_TAGGING_PLANTED), :2] = p4[list(TOP_TAGGING_PLANTED), 1::-1]
    padded = np.zeros((n_jets, 200, 4), dtype=np.float32)
    padded[:, :20] = p4
    return padded, np.arange(n_jets) % 2


def write_top_tagging_raw(raw_dir: Path) -> None:
    # The three raw files as the record stores them: a pandas HDFStore table "table" of
    # the 800 momentum columns and is_signal_new.
    import pandas as pd

    from src.data.top_tagging_datamodule import RAW_COLUMNS

    raw_dir.mkdir(parents=True)
    for name in TOP_TAGGING_SIZES:
        p4, flags = top_tagging_raw(name)
        frame = pd.DataFrame(p4.reshape(len(p4), -1), columns=RAW_COLUMNS)
        frame["is_signal_new"] = flags
        frame.to_hdf(raw_dir / name, key="table", format="table")


@pytest.fixture
def top_tagging_cache(tmp_path: Path) -> Path:
    # The cache root of the synthetic record at N = TOP_TAGGING_N, with the raw files in
    # its raw/, so that nothing is downloaded.
    from src.data.top_tagging_datamodule import convert

    write_top_tagging_raw(tmp_path / "raw")
    convert(tmp_path / "raw", tmp_path, TOP_TAGGING_N, chunk=16)
    return tmp_path
