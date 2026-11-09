"""The top-quark tagging reference data set (Zenodo 2603256, arXiv:1902.09914), converted
once into arrays on disk.

The record holds 1 211 000 training, 403 000 validation and 404 000 test jets: anti-kT
R = 0.8 jets of p_T in [550, 650] GeV and pseudorapidity within (-2, 2), each with the
four-momenta of its leading 200 particle-flow constituents, zero-padded, and the flag
``is_signal_new`` (1 for top, 0 for QCD). The conversion is the sibling repository's
(jet_tagging_gdl), so that both serve the same arrays. It keeps the leading N
constituents by p_T as the nine `COLUMNS`, divides each column by a scale fitted on the
training split as the hls4ml record's loader fits it, and writes ``nconst_<N>/`` under
the cache root:

- ``x_{train,validation,test}.npy``: float32 (jets, N, 9), padded slots all zero;
- ``y_{train,validation,test}.npy``: float32 one-hot (jets, 2), column 0 QCD, 1 top;
- ``meta.json``: the provenance, the counts per split and the scales, written last.

The training and validation splits are pooled and cut into folds, or kept as the record
split them for the main run; the test split is served untouched.
"""

import fcntl
import hashlib
import json
import os
import shutil
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.components.augmentation import (
    MASS_COLUMN,
    append_mass_column,
    compute_eta,
    compute_phi,
    compute_pt,
    rebase_on_kept_axis,
)
from src.data.components.folds import FoldedDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

RECORD_URL = "https://zenodo.org/records/2603256/files/{name}?download=1"
# The md5 of each raw file, from https://zenodo.org/api/records/2603256.
RAW_FILES = {
    "train.h5": "45663819f47c13724f67eb0fd80bfa5c",
    "val.h5": "dca4b7248027618f041f9baa86d360fc",
    "test.h5": "13163479dee30a5fe546e4536cc3d04d",
}
# The datamodule's name of each split, and the raw file holding it.
SPLITS = {"train": "train.h5", "validation": "val.h5", "test": "test.h5"}
MAX_CONSTITUENTS = 200
# The four-momenta of the raw files, constituent by constituent.
RAW_COLUMNS = [
    f"{c}_{i}" for i in range(MAX_CONSTITUENTS) for c in ("E", "PX", "PY", "PZ")
]

COLUMNS = ("px", "py", "pz", "e", "pt", "eta", "phi", "etarel", "phirel")
# Scaled by their maximum, as the hls4ml record scales its non-negative columns; every
# other column lives on both sides of zero and is scaled by half its range.
POSITIVE_COLUMNS = ("pt", "e")
CLASS_NAMES = ("qcd", "top")
NUM_CLASSES = len(CLASS_NAMES)

META = "meta.json"
# The sibling's format, so that either repository reads the other's cache.
FORMAT_VERSION = 1
# Jets per read of a raw file, which bounds the memory of the conversion.
CHUNK = 100_000
# Jets per rebasing chunk, the sibling's.
REBASE_CHUNK = 65536

# One read of a raw file: the four-momenta (jets, 200, 4) as [E, px, py, pz] in GeV,
# and the flags (jets,).
Chunk = tuple[np.ndarray, np.ndarray]


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(url: str, out: Path) -> None:
    # B310 warns of file:// URLs: RECORD_URL is https, and the tests use file:// on
    # purpose.
    with urllib.request.urlopen(url, timeout=600) as response:  # nosec B310
        with open(out, "wb") as handle:
            shutil.copyfileobj(response, handle)


def _fetch(name: str, path: Path) -> None:
    """Download raw file *name* to *path* through a partial file of this process.

    :raises ValueError: if the md5 of the download is not the record's; nothing is left
        at *path*.
    """
    partial = path.with_name(f"{name}.{os.getpid()}.partial")
    log.info(f"Downloading {RECORD_URL.format(name=name)} to {path}")
    try:
        _stream(RECORD_URL.format(name=name), partial)
        if (found := _md5(partial)) != RAW_FILES[name]:
            raise ValueError(f"{name}: md5 {found}, the record's is {RAW_FILES[name]}.")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def download(raw_dir: str | Path) -> None:
    """Download every raw file absent from *raw_dir* or whose md5 is not the record's;
    nothing is written there when all three are intact."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in RAW_FILES.items():
        path = raw_dir / name
        if not (path.is_file() and _md5(path) == md5):
            _fetch(name, path)


def count_jets(path: Path) -> int:
    """The number of jets in raw file *path*, read off the HDF5 metadata alone."""
    with pd.HDFStore(path, mode="r") as store:
        return int(store.get_storer("table").nrows)


def read_chunks(path: Path, chunk: int = CHUNK) -> Iterator[Chunk]:
    """Yield the jets of raw file *path* in their stored order, *chunk* at a time.

    pandas reads the file through PyTables, which it imports on the first read.
    """
    for start in range(0, count_jets(path), chunk):
        frame = pd.read_hdf(path, "table", start=start, stop=start + chunk)
        p4 = frame[RAW_COLUMNS].to_numpy(np.float32).reshape(len(frame), -1, 4)
        yield p4, frame["is_signal_new"].to_numpy()


def _columns(p4: np.ndarray) -> np.ndarray:
    """The nine `COLUMNS` (jets, N, 9) in float64 of the kept constituents *p4*.

    etarel and phirel are measured from the axis of the kept constituents, as the
    datamodule's rebase measures them; every column of a padded slot is zero.
    """
    e, px, py, pz = torch.from_numpy(p4).unbind(dim=-1)
    zero = torch.zeros_like(e)
    angles = (compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py))
    x = torch.stack([px, py, pz, e, *angles, zero, zero], dim=-1) * (e > 0)[..., None]
    return rebase_on_kept_axis(x, COLUMNS).numpy()


def _leading(p4: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """The leading *n* constituents of every jet by p_T, in float64, and the number of
    jets whose stored order the sort changed.

    The sort is stable, so that the padded slots, all at p_T = 0, keep their order and
    move no jet into the count.
    """
    pt = np.hypot(p4[..., 1].astype(np.float64), p4[..., 2])
    order = np.argsort(-pt, axis=1, kind="stable")
    resorted = int((order != np.arange(p4.shape[1])).any(axis=1).sum())
    kept = np.take_along_axis(p4, order[:, :n, None], axis=1)
    return kept.astype(np.float64), resorted


@dataclass
class _Summary:
    """What the conversion records of one split, the multiplicities being those of the
    untruncated jets."""

    n_jets: int = 0
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(2, np.int64))
    jets_resorted: int = 0
    jets_truncated: int = 0
    constituents: int = 0
    max_multiplicity: int = 0
    # Per column, over the kept constituents, padded slots included.
    low: np.ndarray = field(default_factory=lambda: np.full(len(COLUMNS), np.inf))
    high: np.ndarray = field(default_factory=lambda: np.full(len(COLUMNS), -np.inf))

    def add(
        self, p4: np.ndarray, x: np.ndarray, labels: np.ndarray, resorted: int
    ) -> None:
        """Count one chunk: *p4* its raw jets (jets, 200, 4), *x* their stored
        constituents (jets, N, 9) and *labels* their class indices (jets,)."""
        multiplicity = (p4[..., 0] > 0).sum(axis=1)
        self.n_jets += len(x)
        self.class_counts += np.bincount(labels, minlength=2)
        self.jets_resorted += resorted
        self.jets_truncated += int((multiplicity > x.shape[1]).sum())
        self.constituents += int(multiplicity.sum())
        self.max_multiplicity = max(self.max_multiplicity, int(multiplicity.max()))
        self.low = np.minimum(self.low, x.min(axis=(0, 1)))
        self.high = np.maximum(self.high, x.max(axis=(0, 1)))

    def record(self) -> dict[str, object]:
        return {
            "n_jets": self.n_jets,
            "class_counts": dict(zip(CLASS_NAMES, self.class_counts.tolist())),
            "jets_resorted": self.jets_resorted,
            "jets_truncated": self.jets_truncated,
            "mean_multiplicity": self.constituents / self.n_jets,
            "max_multiplicity": self.max_multiplicity,
        }


def _labels(flags: np.ndarray) -> np.ndarray:
    """The class indices of *flags*, which must be 0 (QCD) or 1 (top)."""
    if not np.isin(flags, (0, 1)).all():
        raise ValueError(f"is_signal_new holds {np.unique(flags)}, expected 0 and 1.")
    return flags.astype(np.int64)


def _fill(x: np.ndarray, labels: np.ndarray, chunks: Iterator[Chunk]) -> _Summary:
    """Fill *x* (jets, N, 9) with the kept constituents of *chunks*, unscaled, and
    *labels* (jets,) with their class indices."""
    summary = _Summary()
    for p4, flags in chunks:
        kept, resorted = _leading(p4, x.shape[1])
        rows = slice(summary.n_jets, summary.n_jets + len(p4))
        x[rows], labels[rows] = _columns(kept), _labels(flags)
        summary.add(p4, x[rows], labels[rows], resorted)
    return summary


def _write_split(path: Path, folder: Path, split: str, n: int, chunk: int) -> _Summary:
    """Write *split* from raw file *path* into *folder*, the constituents unscaled, and
    return what was seen."""
    n_jets = count_jets(path)
    shape = (n_jets, n, len(COLUMNS))
    x = np.lib.format.open_memmap(folder / f"x_{split}.npy", "w+", np.float32, shape)
    labels = np.empty(n_jets, dtype=np.int64)
    summary = _fill(x, labels, read_chunks(path, chunk))
    if summary.n_jets != n_jets:
        raise ValueError(f"{path}: read {summary.n_jets} jets of {n_jets}.")
    x.flush()
    np.save(folder / f"y_{split}.npy", np.eye(2, dtype=np.float32)[labels])
    return summary


def _scale(name: str, low: float, high: float) -> tuple[str, float]:
    if name in POSITIVE_COLUMNS:
        return "positive", high
    return "centered", 0.5 * (high - low)


def _fit_scales(train: _Summary) -> dict[str, tuple[str, float]]:
    """One (kind, scale) per column from the training split, as the hls4ml record's
    loader fits them (``fit_physics_norm``)."""
    scales = {
        name: _scale(name, float(low), float(high))
        for name, low, high in zip(COLUMNS, train.low, train.high)
    }
    if not all(scale > 0 for _, scale in scales.values()):
        raise ValueError(f"A column is constant on the training split: {scales}.")
    return scales


def _divide(path: Path, scales: dict[str, tuple[str, float]], chunk: int) -> None:
    """Divide the stored columns of *path* by their scales in place, *chunk* jets at a
    time."""
    x = np.load(path, mmap_mode="r+")
    factors = np.array([scales[name][1] for name in COLUMNS])
    for start in range(0, len(x), chunk):
        x[start : start + chunk] /= factors
    x.flush()


def _write_meta(
    folder: Path,
    n: int,
    summaries: dict[str, _Summary],
    scales: dict[str, tuple[str, float]],
) -> None:
    """Write ``meta.json``: the provenance, the counts of every split and the scales."""
    meta = {
        "format_version": FORMAT_VERSION,
        "source": "https://zenodo.org/records/2603256",
        "files": RAW_FILES,
        "nconstituents": n,
        "columns": list(COLUMNS),
        "class_names": list(CLASS_NAMES),
        "splits": {split: summary.record() for split, summary in summaries.items()},
        "norm_params": scales,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / META).write_text(json.dumps(meta, indent=2) + "\n")


def _build(folder: Path, raw_dir: Path, n: int, chunk: int) -> None:
    """Write the whole cache into *folder*, ``meta.json`` last."""
    summaries = {
        split: _write_split(raw_dir / name, folder, split, n, chunk)
        for split, name in SPLITS.items()
    }
    scales = _fit_scales(summaries["train"])
    for split in SPLITS:
        _divide(folder / f"x_{split}.npy", scales, chunk)
    _write_meta(folder, n, summaries, scales)


def convert(
    raw_dir: str | Path, cache_dir: str | Path, n: int, chunk: int = CHUNK
) -> Path:
    """Convert the raw files in *raw_dir* into ``nconst_<n>/`` under *cache_dir*.

    The cache is written into a partial directory of this process and renamed into place
    once complete, so that a cache with ``meta.json`` is whole; an existing cache is
    never overwritten, and the partial directory is removed on any failure.

    :param chunk: jets per read of a raw file and per division of a stored split.
    :return: the cache directory.
    """
    if not 0 < n <= MAX_CONSTITUENTS:
        raise ValueError(f"nconstituents must lie in 1..{MAX_CONSTITUENTS}, not {n}.")
    target = Path(cache_dir) / f"nconst_{n}"
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    partial.mkdir(parents=True)
    try:
        _build(partial, Path(raw_dir), n, chunk)
        os.rename(partial, target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    log.info(f"Converted the top-tagging record into {target}")
    return target


def read_meta(folder: Path) -> dict:
    """The ``meta.json`` of the cache in *folder*.

    :raises ValueError: if the cache was written in another format.
    """
    meta = json.loads((folder / META).read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{folder} holds a cache of format {meta.get('format_version')}, and this "
            f"code reads format {FORMAT_VERSION}: convert into a fresh cache_dir."
        )
    return meta


def _prepare(cache_dir: Path, raw_dir: Path, n: int) -> None:
    """Download and convert the record unless ``nconst_<n>/`` under *cache_dir* has its
    ``meta.json``, under an exclusive lock on ``<cache_dir>/.lock``."""
    meta = cache_dir / f"nconst_{n}" / META
    if meta.is_file():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    # All the processes of a launch prepare at once: one converts, the others wait.
    with open(cache_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not meta.is_file():
            download(raw_dir)
            convert(raw_dir, cache_dir, n)


class TopTaggingDataModule(FoldedDataModule):
    """Serve the converted top-tagging jets, transformed once when the regime asks, and
    folded.

    Batches carry the nine `COLUMNS` and `MASS_COLUMN`, ``(jets, nconstituents, 10)``,
    with one-hot labels over (qcd, top). Every split is rebased on the axis of the kept
    constituents as it is loaded, as the sibling's datamodule rebases it, so that both
    serve the same rows, and `MASS_COLUMN` is appended to it.

    :param cache_dir: holds one ``nconst_<N>/`` per constituent count, which
        `prepare_data` converts on first use.
    :param raw_dir: the record's three HDF5 files, downloaded into it when absent or
        corrupt; None is ``<cache_dir>/raw``.
    :param nconstituents: how many constituents to keep by leading p_T for each jet.
    :param fold_idx: the fold served, counted from zero; None serves the record's own
        train and validation split, the main run.
    """

    num_classes = NUM_CLASSES

    def __init__(
        self,
        cache_dir: str = "data/top_tagging",
        raw_dir: str | None = None,
        dataset: str = "top_tagging",
        nconstituents: int = 100,
        regime: str = "canonical",
        group: str = "lorentz_iso",
        transform_seed: int = 20260820,
        transform_kwargs: dict[str, float | None] | None = None,
        transformed_dir: str = "data/transformed",
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_folds: int = 5,
        kfold_seed: int = 42,
        fold_idx: int | None = 0,
        teacher_cache: str | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            regime=regime,
            group=group,
            transform_seed=transform_seed,
            transform_kwargs=transform_kwargs,
            transformed_dir=transformed_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            n_folds=n_folds,
            kfold_seed=kfold_seed,
            fold_idx=fold_idx,
            teacher_cache=teacher_cache,
        )
        self.cache_dir = cache_dir
        self.raw_dir = raw_dir or f"{cache_dir}/raw"
        self.nconstituents = nconstituents

    @property
    def cache_folder(self) -> Path:
        return Path(self.cache_dir) / f"nconst_{self.nconstituents}"

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.nconstituents, len(COLUMNS) + 1)

    @property
    def feature_names(self) -> list[str]:
        return [*COLUMNS, MASS_COLUMN]

    @property
    def norm_scales(self) -> dict[str, float] | None:
        """Per column, the factor back to GeV or radians; None until the cache
        exists."""
        if not (self.cache_folder / META).is_file():
            return None
        params = read_meta(self.cache_folder)["norm_params"]
        return {name: scale for name, (_, scale) in params.items()}

    def prepare_data(self) -> None:
        """Download the raw files and convert them, one process at a time, unless the
        cache exists, whose format is then checked."""
        _prepare(Path(self.cache_dir), Path(self.raw_dir), self.nconstituents)
        read_meta(self.cache_folder)
        log.info(f"Top-tagging cache {self.cache_folder}")

    def load_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The record's training and validation splits, pooled, in that order."""
        return self._load("train", "validation")

    def main_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The record's own split: its training rows, then its validation rows."""
        n_train = read_meta(self.cache_folder)["splits"]["train"]["n_jets"]
        n_pool = len(self.split("pool")[0])
        return torch.arange(n_train), torch.arange(n_train, n_pool)

    def load_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load("test")

    def _load(self, *splits: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The jets of *splits*, concatenated in memory, rebased on the kept axis and
        with `MASS_COLUMN` appended, and their one-hot labels."""
        folder, scales = self.cache_folder, self.norm_scales
        stored = [np.load(folder / f"x_{s}.npy", mmap_mode="r") for s in splits]
        x = torch.from_numpy(np.concatenate(stored))
        # Chunked from each split's first row, as the sibling rebases them: torch's
        # vectorised and scalar kernels of log and atan2 may round apart in the last
        # place, and the chunk bounds set which elements take which kernel.
        for part in x.split([len(a) for a in stored]):
            for chunk in part.split(REBASE_CHUNK):
                chunk.copy_(rebase_on_kept_axis(chunk, COLUMNS, scales))
        y = np.concatenate([np.load(folder / f"y_{s}.npy") for s in splits])
        return append_mass_column(x, COLUMNS, scales), torch.from_numpy(y)
