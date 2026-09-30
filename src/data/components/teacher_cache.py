"""The teacher's logits and hints, frozen once per (dataset, regime, fold).

A distillation run reads the teacher from this file rather than holding the teacher in
memory beside the student, which is what lets the student train alone. The rows of the
pool arrays are in the pool's stored order, so the fold indices that cut the inputs cut
them too, and the alignment needs no key of its own.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.components.transformed import savez_atomic

ARRAYS = ("pool_logits", "pool_hints", "test_logits", "test_hints")


@dataclass
class TeacherCache:
    """Logits (rows, classes) and hints (rows, hint_dim) of both splits, float32."""

    pool_logits: np.ndarray
    pool_hints: np.ndarray
    test_logits: np.ndarray
    test_hints: np.ndarray
    meta: dict


def write_cache(path: str | Path, cache: TeacherCache) -> None:
    """Write *cache* as one npz, its meta as a JSON string entry beside the arrays."""
    arrays = {
        name: np.asarray(getattr(cache, name), dtype=np.float32) for name in ARRAYS
    }
    savez_atomic(Path(path), meta=json.dumps(cache.meta), **arrays)


def read_cache(path: str | Path, expect: dict) -> TeacherCache:
    """Read the cache at *path*, refusing one written for another run.

    :param expect: keys of the meta the caller depends on, such as the dataset, the
        regime and the fold; a value that differs is a misalignment, not a warning.
    :raises ValueError: if any key of *expect* differs from the stored meta.
    """
    stored = np.load(path)
    meta = json.loads(str(stored["meta"].item()))
    _check(meta, expect, path)
    return TeacherCache(*(stored[name] for name in ARRAYS), meta=meta)


def _check(meta: dict, expect: dict, path: str | Path) -> None:
    wrong = {
        key: (meta.get(key), value)
        for key, value in expect.items()
        if meta.get(key) != value
    }
    if wrong:
        raise ValueError(
            f"The teacher cache {path} does not match this run; "
            f"{{key: (cached, wanted)}} = {wrong}."
        )
