"""Per-sample arrays of one evaluation, for the analyses that the means in results.csv
cannot serve (the decomposition's magnitude bins and teacher-error split)."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.components.transformed import savez_atomic


def write_samples(
    path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]
) -> None:
    """Write *arrays* and the JSON *meta* that identifies them to one ``.npz``.

    Through a temporary file and a rename, so a reader never meets a truncated dump.
    """
    savez_atomic(path, meta=np.array(json.dumps(meta, default=str)), **arrays)


def read_samples(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files if name != "meta"}
        return arrays, json.loads(str(stored["meta"]))
