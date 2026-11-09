# Import one teacher checkpoint of the sibling repository as a finished teacher run.

import hashlib
from pathlib import Path
from typing import Any

import hydra
import lightning
import rootutils
import torch
from lightning import LightningModule
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import fold_value, register_resolvers

# Before Hydra composes: the configs interpolate ${len:}, ${short_hash:}, ${fold_idx:},
# ${fold_label:} and ${regime_of_teacher:}.
register_resolvers()

from src.train import _already_done, _check_fold
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.checkpoints import Manifest, config_hash, write_manifest

log = RankedLogger(__name__, rank_zero_only=True)


def map_state_dict(
    state: dict[str, torch.Tensor], drop: list[str], renames: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Return the sibling's *state* under this repository's keys.

    The keys in *drop* are removed; a key starting with a key of *renames* then takes
    that entry's value as its prefix instead.
    """
    kept = {key: value for key, value in state.items() if key not in drop}
    return {_renamed(key, renames): value for key, value in kept.items()}


def _renamed(key: str, renames: dict[str, str]) -> str:
    old = next((prefix for prefix in renames if key.startswith(prefix)), None)
    return key if old is None else renames[old] + key[len(old) :]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_score(source: dict[str, Any]) -> float | None:
    """The score by which the sibling's ModelCheckpoint selected this checkpoint."""
    states = source.get("callbacks", {})
    scores = [s.get("best_model_score") for k, s in states.items() if "Checkpoint" in k]
    return None if not scores or scores[0] is None else float(scores[0])


def _imported(cfg: DictConfig, state: dict[str, torch.Tensor]) -> LightningModule:
    """The composed algorithm holding the sibling's weights, loaded strictly."""
    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    drop, renames = list(cfg.source_drop_keys), dict(cfg.source_key_renames)
    algorithm.load_state_dict(map_state_dict(state, drop, renames), strict=True)
    return algorithm


def _save(algorithm: LightningModule, source: dict[str, Any], path: Path) -> None:
    """Write the checkpoint ``load_from_checkpoint`` rebuilds the module from, the
    hyperparameters being the plain dicts the module keeps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": algorithm.state_dict(),
        "hyper_parameters": dict(algorithm.hparams),
        "pytorch-lightning_version": lightning.__version__,
        "epoch": source.get("epoch"),
        "global_step": source.get("global_step"),
    }
    torch.save(checkpoint, path)
    log.info(f"Wrote {path}")


def _manifest(
    cfg: DictConfig, source: dict[str, Any], checkpoint: Path, digest: str
) -> Manifest:
    """The manifest train.py would write, plus the sibling checkpoint behind it."""
    source_path = Path(cfg.source_ckpt).resolve()
    return Manifest(
        stage=str(cfg.stage),
        fold=fold_value(cfg.fold),
        n_folds=int(cfg.data.n_folds),
        checkpoint_path=str(checkpoint.resolve()),
        hydra_output_dir=str(cfg.paths.output_dir),
        experiment_name=str(cfg.experiment_name),
        config_hash=digest,
        # Up to the imported checkpoint; the sibling's run may have gone on past it.
        epochs_run=int(source["epoch"]) + 1,
        best_metric=_best_score(source),
        source_checkpoint=str(source_path),
        source_sha256=_sha256(source_path),
    )


@task_wrapper
def import_teacher(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import ``source_ckpt`` as the teacher run of this experiment and fold.

    :return: no metrics, and the objects the run was built from; both empty when the run
        directory already holds a run of this configuration.
    """
    _check_fold(cfg)
    digest = config_hash(cfg)
    if _already_done(cfg, digest):
        return {}, {}
    source = torch.load(cfg.source_ckpt, map_location="cpu", weights_only=False)
    algorithm = _imported(cfg, source["state_dict"])
    checkpoint = Path(cfg.paths.output_dir) / "checkpoints" / Path(cfg.source_ckpt).name
    _save(algorithm, source, checkpoint)
    write_manifest(_manifest(cfg, source, checkpoint, digest), cfg.paths.output_dir)
    return {}, {"cfg": cfg, "algorithm": algorithm}


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="import_teacher.yaml"
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    import_teacher(cfg)


if __name__ == "__main__":
    main()
