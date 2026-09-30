"""Custom ``${...}`` resolvers for the Hydra configs."""

import hashlib

from omegaconf import OmegaConf


def short_hash(value: str, length: int = 12) -> str:
    """The first *length* hex characters of the SHA-1 of *value*; names a run."""
    # str(): under --cfg job ${hydra:runtime.output_dir} is None.
    return hashlib.sha1(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[
        :length
    ]


def fold_idx(fold: int) -> int:
    """The zero-based index of the one-based ``fold`` knob."""
    return int(fold) - 1


def register_resolvers() -> None:
    """Register every resolver; ``replace=True`` makes a second call harmless."""
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
    OmegaConf.register_new_resolver("fold_idx", fold_idx, replace=True)
