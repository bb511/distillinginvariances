#!/usr/bin/env python3
"""Export a trained quantised student for scripts/synthesis/synthesise.py.

The run directory of an ``hgq`` stage holds its composed config (``.hydra/config.yaml``)
and its manifest. The datamodule is rebuilt from that config and the checkpoint loaded
with JAX on the CPU; the integer bits of every wrapping quantiser are traced on the
run's training split (the main run's for ``fold=null``), and the test split is
predicted. ``--out``, by default ``<run dir>/export``, receives ``model.keras``,
``inputs.npz`` (the words the port reads), ``export.json`` and ``reference.npz`` (the
logits and the labels)::

    python scripts/synthesis/export_hgq.py \\
        --run-dir logs/train/jets/deepsets/canonical/hgq/main/lr0.003 --out <dir>
"""

import argparse
import logging
import os
from pathlib import Path

# Before anything imports JAX: XLA's float64 2**f is inexact on a GPU, so the reference
# logits would not match alkaid bit for bit.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import rootutils
from hydra.utils import get_class, instantiate
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, read at import.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.checkpoints import checkpoint_of
from src.utils.omegaconf import register_resolvers
from src.utils.pylogger import RankedLogger

# The saved config interpolates ${fold_idx:} and ${fold_label:}.
register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="an hgq run")
    parser.add_argument("--ckpt", help="None: the checkpoint the manifest names")
    parser.add_argument("--out", type=Path, help="None: <run dir>/export")
    return parser.parse_args(argv)


def _datamodule(cfg: DictConfig) -> LightningDataModule:
    """The run's datamodule, set up: the export runs without a Trainer."""
    datamodule: LightningDataModule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def _load(cfg: DictConfig, checkpoint: str) -> LightningModule:
    """The run's quantised module, with JAX on the CPU whatever it trained on."""
    module_class = get_class(cfg.algorithm._target_)
    log.info(f"Loading {module_class.__name__} from {checkpoint}")
    return module_class.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        jax_platform="cpu",
        jax_devices=1,
    )


def main(argv: list[str] | None = None) -> int:
    """Export the run the command line names; return the exit status."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    checkpoint = args.ckpt or checkpoint_of(args.run_dir)
    datamodule, module = _datamodule(cfg), _load(cfg, checkpoint)
    module.bind(datamodule)
    provenance = {"row": str(cfg.run_dir), "ckpt_path": str(checkpoint)}
    meta = module.export(args.out or args.run_dir / "export", datamodule, provenance)
    log.info(f"{meta['ebops']} EBOPs, test accuracy {meta['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
