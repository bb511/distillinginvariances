"""Manifest of the checkpoint one training run produced."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

MANIFEST_VERSION = 2
MANIFEST_FILENAME = "checkpoint_manifest.json"

# The config subtrees and scalars that define a run scientifically, including the study-level
# knobs the algorithm and data nodes reach through an interpolation, which an unresolved hash
# would otherwise see as "${...}". trainer.devices, the logger and the tags are left out: they
# change between launches of one run.
HASHED_KEYS = (
    "data",
    "teacher_net",
    "student_net",
    "algorithm",
    "seed",
    "fold",
    "stage",
    "monitor",
    "monitor_mode",
    "patience",
    "teacher_lr",
    "teacher_batch_size",
    "teacher_optimizer",
    "teacher_scheduler",
    "hint_lr",
    "relational_batch_size",
    "relational_scheduler",
)
HASHED_TRAINER_KEYS = ("max_epochs", "precision", "gradient_clip_val")


@dataclass
class Manifest:
    """What eval.py and a downstream stage need to find one run's checkpoint."""

    stage: str
    fold: int | str  # 1..n_folds, or MAIN_FOLD for the main run
    n_folds: int
    checkpoint_path: str
    hydra_output_dir: str
    experiment_name: str
    config_hash: str
    epochs_run: int
    best_metric: float | None = None
    mlflow_run_id: str | None = None
    # Set by src/import_teacher.py alone: the sibling checkpoint the run was imported from.
    source_checkpoint: str | None = None
    source_sha256: str | None = None
    version: int = MANIFEST_VERSION


def config_hash(cfg: DictConfig) -> str:
    """Hash of the knobs that make a run; unresolved, so no live HydraConfig is
    needed."""
    picked = {key: OmegaConf.select(cfg, key) for key in HASHED_KEYS}
    picked["trainer"] = {
        k: OmegaConf.select(cfg, f"trainer.{k}") for k in HASHED_TRAINER_KEYS
    }
    text = OmegaConf.to_yaml(OmegaConf.create(picked), resolve=False)
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def write_manifest(manifest: Manifest, output_dir: str | Path) -> Path:
    """Write *manifest* to ``<output_dir>/checkpoint_manifest.json``."""
    path = Path(output_dir) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2))
    log.info(f"Wrote {path}")
    return path


def read_manifest(path: str | Path) -> Manifest:
    """Load a manifest; a version mismatch raises rather than dropping fields
    silently."""
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"{path} has manifest version {payload.get('version')!r}, this code reads "
            f"{MANIFEST_VERSION}. Re-run the training that wrote it."
        )
    return _rebased(Manifest(**payload), Path(path).parent)


def _rebased(manifest: Manifest, run_dir: Path) -> Manifest:
    """The manifest with its checkpoint found under *run_dir* when the recorded absolute
    path is absent, as for a run directory copied to another machine (olgpu to NGT)."""
    ckpt, written_in = Path(manifest.checkpoint_path), manifest.hydra_output_dir
    if ckpt.exists() or not ckpt.is_relative_to(written_in):
        return manifest
    manifest.checkpoint_path = str(run_dir / ckpt.relative_to(written_in))
    return manifest


def existing_manifest(output_dir: str | Path) -> Manifest | None:
    """The manifest already in *output_dir*, or None."""
    path = Path(output_dir) / MANIFEST_FILENAME
    return read_manifest(path) if path.is_file() else None


def checkpoint_of(run_dir: str | Path) -> str:
    """The checkpoint a finished run directory names."""
    return read_manifest(Path(run_dir) / MANIFEST_FILENAME).checkpoint_path


def best_or_last_checkpoint(trainer: Trainer) -> str:
    """Absolute path of the best checkpoint by the monitored metric, or the last one.

    :raises RuntimeError: when the run wrote no checkpoint at all.
    """
    callback = getattr(trainer, "checkpoint_callback", None)
    for candidate in (
        getattr(callback, "best_model_path", ""),
        getattr(callback, "last_model_path", ""),
    ):
        if candidate:
            return str(Path(candidate).resolve())
    raise RuntimeError("The run produced no checkpoint; enable callbacks=default.")
