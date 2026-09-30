"""The one-shot transform of a whole split, and its cache on disk.

The transformed regime draws once per row, before any fold is cut, so every fold of a
dataset sees the same transformed rows and a run may be compared against the canonical
one row by row. The draw costs minutes on the jets pool, so it is memoised under
``<transformed_dir>/<dataset>/<group>/seed<seed>/`` and rebuilt whenever the key it was
drawn under changes.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from src.data.components.augmentation import apply_group
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Rows per chunk. Chunking bounds the peak memory of the draw; it is part of the
# reproducibility contract, since the RNG is consumed chunk by chunk.
TRANSFORM_CHUNK = 65536


def transformed_split(
    x: torch.Tensor,
    group: str,
    seed: int,
    feature_names: list[str] | None = None,
    norm_scales: dict[str, float] | None = None,
    chunk: int = TRANSFORM_CHUNK,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transform every row of *x* once under *group*, reproducibly.

    The draw runs inside a forked RNG state seeded with *seed*, so it depends on nothing
    but the arguments and leaves the ambient RNG of the caller untouched.

    :param kwargs: the knobs of `~src.data.components.augmentation.apply_group`.
    :return: the transformed rows and, for ``lorentz_iso`` alone, the matrices that
        produced them, shape (rows, 4, 4) on [E, px, py, pz] in GeV.
    """
    rows, params = [], []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for part in x.split(chunk):
            transformed, drawn = apply_group(
                group, part, feature_names, norm_scales, **kwargs
            )
            rows.append(transformed)
            params.append(drawn)
    matrices = torch.cat(params) if group == "lorentz_iso" else None
    return torch.cat(rows), matrices


def cached_transform(
    directory: str,
    key: dict,
    builder: Callable[[], tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the transformed split of *key*, building and storing it when absent.

    :param key: what the stored draw must agree on, ``key["split"]`` naming the file.
    """
    path = Path(directory)
    arrays, meta = path / f"{key['split']}.npz", path / f"{key['split']}.meta.json"
    cached = _read(arrays, meta, key)
    if cached is not None:
        return cached
    x, matrices = builder()
    _write(arrays, meta, key, x, matrices)
    return x, matrices


def _read(
    arrays: Path, meta: Path, key: dict
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """The stored draw, or None when it is absent or was drawn under another key."""
    if not (arrays.exists() and meta.exists()):
        return None
    if json.loads(meta.read_text()) != key:
        log.info(f"Rebuilding {arrays}: it was drawn under a different key.")
        return None
    stored = np.load(arrays)
    matrices = stored["matrices"] if "matrices" in stored else None
    return (
        torch.from_numpy(stored["x"]),
        None if matrices is None else torch.from_numpy(matrices),
    )


def savez_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an npz through a temporary file and one rename.

    The caches are shared by every fold and every pod, and np.savez writes in place, so
    a process that starts while another is still writing would read a truncated file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _write(
    arrays: Path, meta: Path, key: dict, x: torch.Tensor, matrices: torch.Tensor | None
) -> None:
    stored = {"x": x.numpy()}
    if matrices is not None:
        stored["matrices"] = matrices.numpy()
    savez_atomic(arrays, **stored)
    meta.write_text(json.dumps(key))
    log.info(f"Stored the transformed split at {arrays}.")
