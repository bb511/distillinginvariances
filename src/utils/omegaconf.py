"""Custom ``${...}`` resolvers for the Hydra configs, and the fold value the entry
points record."""

import hashlib

from omegaconf import OmegaConf

from src.vocabulary import MAIN_FOLD, TEACHER_REGIME, regime_tag


def short_hash(value: str, length: int = 12) -> str:
    """The first *length* hex characters of the SHA-1 of *value*; names a run."""
    # str(): under --cfg job ${hydra:runtime.output_dir} is None.
    return hashlib.sha1(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[
        :length
    ]


def fold_idx(fold: int | None) -> int | None:
    """The zero-based index of the one-based ``fold`` knob; None for the main run."""
    return None if fold is None else int(fold) - 1


def fold_label(fold: int | None, default: str) -> str:
    """Return ``fold_<i>`` for fold *fold*, counted from one, or *default* for main.

    Names a run after the one ``fold`` knob of ``train.yaml``, in its run directory and
    its teacher cache.
    """
    return default if fold is None else f"fold_{int(fold)}"


def fold_value(fold: int | None) -> int | str:
    """The fold as the manifest, the cache meta and the result rows record it."""
    return MAIN_FOLD if fold is None else int(fold)


def regime_of_teacher(regime: str) -> str:
    """The regime whose teacher run and cache a run of *regime* distils from."""
    return TEACHER_REGIME[regime]


def register_resolvers() -> None:
    """Register every resolver; ``replace=True`` makes a second call harmless."""
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
    OmegaConf.register_new_resolver("fold_idx", fold_idx, replace=True)
    OmegaConf.register_new_resolver("fold_label", fold_label, replace=True)
    OmegaConf.register_new_resolver(
        "regime_of_teacher", regime_of_teacher, replace=True
    )
    OmegaConf.register_new_resolver("regime_tag", regime_tag, replace=True)
